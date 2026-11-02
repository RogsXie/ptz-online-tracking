#include "PTZArena.h"
#include "Actor/FusionCameraActor.h"
#include "BPFunctionLib/SensorBPLib.h"
#include "Camera/CameraActor.h"
#include "Camera/CameraComponent.h"
#include "Components/SceneComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Components/DirectionalLightComponent.h"
#include "Components/SkyLightComponent.h"
#include "Components/ExponentialHeightFogComponent.h"
#include "Engine/World.h"
#include "Engine/GameViewportClient.h"
#include "Engine/Canvas.h"
#include "Engine/DirectionalLight.h"
#include "Engine/SkyLight.h"
#include "Engine/ExponentialHeightFog.h"
#include "EngineUtils.h"
#include "GameFramework/PlayerController.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Materials/MaterialInterface.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/Guid.h"
#include "LitCamSensor.h"
#include "TextureResource.h"
#include "RHIGPUReadback.h"
#include "RHISurfaceDataConversion.h"
#include "RenderingThread.h"
#include "Serialization/JsonSerializer.h"
#include "Server/UnrealcvServer.h"
#include "UObject/ConstructorHelpers.h"

struct FPTZPendingGPUFrame
{
    TUniquePtr<FRHIGPUTextureReadback> Readback;
    TSharedPtr<FJsonObject> Metadata;
    EPixelFormat Format = PF_Unknown;
    TAtomic<bool> Submitted{false};
    uint64 Sequence = 0;
    double CaptureTime = 0;
};

namespace
{
TArray<TSharedPtr<FJsonValue>> Numbers(std::initializer_list<double> Values)
{
    TArray<TSharedPtr<FJsonValue>> Out;
    for (double V : Values) Out.Add(MakeShared<FJsonValueNumber>(V));
    return Out;
}
TSharedPtr<FJsonObject> Object(const TSharedPtr<FJsonObject>& J, const TCHAR* Key)
{
    if (!J) return MakeShared<FJsonObject>();
    const TSharedPtr<FJsonObject>* Found = nullptr;
    return J->TryGetObjectField(Key, Found) ? *Found : MakeShared<FJsonObject>();
}
double Number(const TSharedPtr<FJsonObject>& J, const TCHAR* Key, double Default)
{
    double V;
    return J && J->TryGetNumberField(Key, V) && FMath::IsFinite(V) ? V : Default;
}
FString String(const TSharedPtr<FJsonObject>& J, const TCHAR* Key, const FString& Default)
{
    FString V;
    return J && J->TryGetStringField(Key, V) ? V : Default;
}
FVector Vector(const TSharedPtr<FJsonObject>& J, const TCHAR* Key, FVector Default)
{
    const TArray<TSharedPtr<FJsonValue>>* Values = nullptr;
    if (!J || !J->TryGetArrayField(Key, Values) || Values->Num() != 3) return Default;
    for (int I = 0; I < 3; ++I)
    {
        double V;
        if (!(*Values)[I]->TryGetNumber(V) || !FMath::IsFinite(V)) return Default;
        Default[I] = V;
    }
    return Default;
}
FVector2D Pair(const TSharedPtr<FJsonObject>& J, const TCHAR* Key, FVector2D Default)
{
    const TArray<TSharedPtr<FJsonValue>>* Values = nullptr;
    if (!J || !J->TryGetArrayField(Key, Values) || Values->Num() != 2) return Default;
    const double A = (*Values)[0]->AsNumber(), B = (*Values)[1]->AsNumber();
    return FMath::IsFinite(A) && FMath::IsFinite(B) && A <= B ? FVector2D(A, B) : Default;
}
FLinearColor Color(const TSharedPtr<FJsonObject>& J, const TCHAR* Key, FLinearColor Default)
{
    const FVector V = Vector(J, Key, FVector(Default.R, Default.G, Default.B));
    return FLinearColor(FMath::Clamp(V.X, 0.0, 1.0), FMath::Clamp(V.Y, 0.0, 1.0), FMath::Clamp(V.Z, 0.0, 1.0));
}
// Analytic piecewise-constant speed integration. Zero pauses and negative reverses.
double IntegratedTime(const TSharedPtr<FJsonObject>& Path, double Time)
{
    const double Loop = Number(Path, TEXT("loop_period_s"), 0);
    if (Loop > 0) Time = FMath::Fmod(FMath::Max(0.0, Time), Loop);
    const TArray<TSharedPtr<FJsonValue>>* Profile = nullptr;
    if (!Path->TryGetArrayField(TEXT("speed_profile"), Profile)) return Time;
    double Result = 0, Previous = 0, Speed = 1;
    for (const auto& Value : *Profile)
    {
        const auto& Row = Value->AsArray();
        if (Row.Num() != 2) continue;
        const double Boundary = FMath::Max(0.0, Row[0]->AsNumber());
        if (Boundary > Time) break;
        Result += FMath::Max(0.0, Boundary - Previous) * Speed;
        Previous = Boundary;
        Speed = Row[1]->AsNumber();
    }
    return Result + FMath::Max(0.0, Time - Previous) * Speed;
}
}

APTZArena::APTZArena()
{
    PrimaryActorTick.bCanEverTick = true;
    RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("ArenaRoot"));
    static ConstructorHelpers::FObjectFinder<UStaticMesh> Cube(TEXT("/Engine/BasicShapes/Cube.Cube"));
    static ConstructorHelpers::FObjectFinder<UStaticMesh> Sphere(TEXT("/Engine/BasicShapes/Sphere.Sphere"));
    static ConstructorHelpers::FObjectFinder<UMaterialInterface> Material(TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
    CubeMesh = Cube.Object;
    SphereMesh = Sphere.Object;
    ShapeMaterial = Material.Object;
    auto Preview = [&](const TCHAR* Name, FVector Position, FVector Scale)
    {
        UStaticMeshComponent* Part = CreateDefaultSubobject<UStaticMeshComponent>(Name);
        Part->SetupAttachment(RootComponent);
        Part->SetStaticMesh(CubeMesh);
        Part->SetRelativeLocation(Position);
        Part->SetRelativeScale3D(Scale);
        Part->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        if (ShapeMaterial) Part->SetMaterial(0, ShapeMaterial);
        PreviewParts.Add(Part);
    };
    Preview(TEXT("PreviewGround"), FVector(3000, 0, -25), FVector(220, 100, 0.5));
    Preview(TEXT("PreviewRoad"), FVector(2600, 0, 2), FVector(100, 35, 0.03));
    for (int I = 0; I < 3; ++I)
    {
        Preview(*FString::Printf(TEXT("PreviewCarBody%d"), I), FVector(1700 + I * 650, -1000 + I * 950, 83), FVector(4, 1.9, 0.7));
        Preview(*FString::Printf(TEXT("PreviewCarRoof%d"), I), FVector(1680 + I * 650, -1000 + I * 950, 142), FVector(1.9, 1.55, 0.65));
        for (int X : {-1, 1}) for (int Y : {-1, 1})
            Preview(*FString::Printf(TEXT("PreviewCarWheel%d_%d_%d"), I, X, Y), FVector(1700 + I * 650 + X * 125, -1000 + I * 950 + Y * 99, 44), FVector(0.65, 0.3, 0.65));
    }
    Preview(TEXT("PreviewBuilding"), FVector(300, -1800, 300), FVector(6, 9, 6));
    Preview(TEXT("PreviewWall"), FVector(2500, 2400, 220), FVector(18, 1.2, 4.4));
}

void APTZArena::OnConstruction(const FTransform& Transform)
{
    Super::OnConstruction(Transform);
    for (UStaticMeshComponent* Part : PreviewParts)
    {
        if (!Part || !ShapeMaterial) continue;
        const FString N = Part->GetName();
        FLinearColor C(0.42f, 0.44f, 0.46f);
        if (N.Contains(TEXT("Body"))) C = FLinearColor(0.9f, 0.018f, 0.012f);
        else if (N.Contains(TEXT("Roof"))) C = FLinearColor(0.035f, 0.06f, 0.08f);
        else if (N.Contains(TEXT("Wheel"))) C = FLinearColor(0.01f, 0.01f, 0.01f);
        else if (N.Contains(TEXT("Ground"))) C = FLinearColor(0.26f, 0.30f, 0.26f);
        else if (N.Contains(TEXT("Road"))) C = FLinearColor(0.09f, 0.10f, 0.11f);
        auto* M = UMaterialInstanceDynamic::Create(ShapeMaterial, Part);
        M->SetVectorParameterValue(TEXT("Color"), C);
        Part->SetMaterial(0, M);
    }
}

UStaticMeshComponent* APTZArena::AddPart(AActor* PartOwner, const TCHAR* Name,
    UStaticMesh* Mesh, FVector Location, FVector Scale, FLinearColor C)
{
    auto* Part = NewObject<UStaticMeshComponent>(PartOwner, FName(Name));
    PartOwner->AddInstanceComponent(Part);
    Part->SetStaticMesh(Mesh);
    Part->SetMobility(EComponentMobility::Movable);
    Part->SetCollisionEnabled(ECollisionEnabled::QueryOnly);
    Part->SetCollisionResponseToAllChannels(ECR_Ignore);
    Part->SetCollisionResponseToChannel(ECC_Visibility, ECR_Block);
    Part->SetupAttachment(PartOwner->GetRootComponent());
    Part->SetRelativeLocation(Location);
    Part->SetRelativeScale3D(Scale);
    if (ShapeMaterial)
    {
        auto* M = UMaterialInstanceDynamic::Create(ShapeMaterial, Part);
        M->SetVectorParameterValue(TEXT("Color"), C);
        Part->SetMaterial(0, M);
    }
    Part->RegisterComponent();
    return Part;
}

AActor* APTZArena::SpawnSceneActor(const FString& Name)
{
    FActorSpawnParameters Params;
    Params.Name = FName(Name);
    Params.NameMode = FActorSpawnParameters::ESpawnActorNameMode::Requested;
    AActor* A = GetWorld()->SpawnActor<AActor>(Params);
    auto* Root = NewObject<USceneComponent>(A, TEXT("SceneRoot"));
    A->AddInstanceComponent(Root);
    A->SetRootComponent(Root);
    Root->SetMobility(EComponentMobility::Movable);
    Root->RegisterComponent();
    SceneActors.Add(A);
    return A;
}

void APTZArena::BeginPlay()
{
    Super::BeginPlay();
    for (UStaticMeshComponent* Part : PreviewParts) if (Part) Part->SetHiddenInGame(true);
    // Editor context stays visible while authoring, but cannot obstruct RGB or GT rays.
    // Hidden-in-game alone does NOT disable physics/query collision in uncooked -game.
    for (TActorIterator<AActor> It(GetWorld()); It; ++It)
        if (It->Tags.Contains(TEXT("PTZ_EditorContext"))) It->SetActorEnableCollision(false);
    FActorSpawnParameters Params;
    Params.Name = TEXT("PTZ_CaptureCamera");
    CaptureCamera = GetWorld()->SpawnActor<AFusionCameraActor>(Params);
    Params.Name = TEXT("PTZ_ViewportCamera");
    ViewCamera = GetWorld()->SpawnActor<ACameraActor>(Params);
    Sensor()->SetFilmSize(960, 540);
    FString Error;
    if (!LoadScenario(ScenarioFile, Error))
    {
        UE_LOG(LogTemp, Error, TEXT("PTZ default scenario unavailable: %s"), *Error);
        ScenarioConfig = MakeShared<FJsonObject>();
        ScenarioConfig->SetStringField(TEXT("name"), TEXT("fallback_static"));
        auto Target = MakeShared<FJsonObject>();
        Target->SetStringField(TEXT("id"), TEXT("car_0"));
        auto Path = MakeShared<FJsonObject>();
        Path->SetStringField(TEXT("type"), TEXT("static"));
        Path->SetArrayField(TEXT("origin_cm"), Numbers({1800, 500, 8}));
        Target->SetObjectField(TEXT("path"), Path);
        ScenarioConfig->SetArrayField(TEXT("targets"), {MakeShared<FJsonValueObject>(Target)});
        ResetEpisode(0);
    }
    RegisterCommands();
    SetPreview(true);
    if (auto* PC = GetWorld()->GetFirstPlayerController()) PC->SetViewTarget(ViewCamera);
    UE_LOG(LogTemp, Display, TEXT("PTZ UE replacement arena ready: seeded JSON, fixed-camera motion preview; reset/step are synchronous"));
}

UFusionCamSensor* APTZArena::Sensor() const
{
    return CaptureCamera ? CaptureCamera->GetSensors()[0] : nullptr;
}

bool APTZArena::LoadScenario(const FString& Filename, FString& Error)
{
    FString Path = Filename;
    if (FPaths::IsRelative(Path)) Path = FPaths::ConvertRelativePathToFull(FPaths::ProjectDir(), Path);
    FString Text;
    TSharedPtr<FJsonObject> Parsed;
    if (!FFileHelper::LoadFileToString(Text, *Path) || !FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text), Parsed) || !Parsed)
    { Error = TEXT("Cannot read valid JSON scenario: ") + Path; return false; }
    const TArray<TSharedPtr<FJsonValue>>* JsonTargets = nullptr;
    if (!Parsed->TryGetArrayField(TEXT("targets"), JsonTargets) || JsonTargets->Num() < 1 || JsonTargets->Num() > 32)
    { Error = TEXT("Scenario requires 1..32 targets"); return false; }
    TSet<FString> Ids;
    for (const auto& V : *JsonTargets)
    {
        const auto T = V->AsObject();
        const auto P = Object(T, TEXT("path"));
        const FString Id = String(T, TEXT("id"), TEXT("")), Type = String(P, TEXT("type"), TEXT(""));
        if (!T || Id.IsEmpty() || Ids.Contains(Id) || !(Type == TEXT("static") || Type == TEXT("line") || Type == TEXT("figure8") || Type == TEXT("waypoints") || Type == TEXT("ellipse")))
        { Error = TEXT("Targets need unique ids and supported paths"); return false; }
        const TArray<TSharedPtr<FJsonValue>>* Profile = nullptr;
        if (P->TryGetArrayField(TEXT("speed_profile"), Profile))
        {
            double PreviousBoundary = -1;
            for (const auto& RowValue : *Profile)
            {
                const TArray<TSharedPtr<FJsonValue>>* Row = nullptr;
                double Boundary = 0, Speed = 0;
                if (!RowValue->TryGetArray(Row) || Row->Num() != 2 ||
                    !(*Row)[0]->TryGetNumber(Boundary) || !(*Row)[1]->TryGetNumber(Speed) ||
                    !FMath::IsFinite(Boundary) || !FMath::IsFinite(Speed) || Boundary < 0 || Boundary <= PreviousBoundary)
                { Error = TEXT("speed_profile must contain finite [time,speed] pairs in strictly increasing nonnegative time order"); return false; }
                PreviousBoundary = Boundary;
            }
        }
        if (Type == TEXT("waypoints"))
        {
            const TArray<TSharedPtr<FJsonValue>>* Points = nullptr;
            if (!P->TryGetArrayField(TEXT("waypoints_cm"), Points) || Points->Num() < 2)
            { Error = TEXT("waypoints path needs at least two 3D points"); return false; }
            for (const auto& Point : *Points)
            {
                const TArray<TSharedPtr<FJsonValue>>* Row = nullptr;
                if (!Point->TryGetArray(Row) || Row->Num() != 3)
                { Error = TEXT("Each waypoint must be a 3D numeric array"); return false; }
                for (const auto& CoordinateValue : *Row)
                {
                    double Coordinate;
                    if (!CoordinateValue->TryGetNumber(Coordinate) || !FMath::IsFinite(Coordinate))
                    { Error = TEXT("Waypoint coordinates must be finite numbers"); return false; }
                }
            }
        }
        Ids.Add(Id);
    }
    ScenarioFile = Path;
    ScenarioConfig = Parsed;
    ResetEpisode(EpisodeSeed);
    return true;
}

void APTZArena::BuildScenario(FRandomStream& Random)
{
    for (AActor* A : SceneActors) if (IsValid(A)) A->Destroy();
    SceneActors.Empty();
    Targets.Empty();
    ResolvedConfig = MakeShared<FJsonObject>();
    ResolvedConfig->SetObjectField(TEXT("source"), ScenarioConfig);
    ResolvedConfig->SetNumberField(TEXT("seed"), EpisodeSeed);
    const auto Rand = Object(ScenarioConfig, TEXT("randomization"));
    const double Jitter = FMath::Max(0.0, Number(Rand, TEXT("position_jitter_cm"), 0));
    const FVector2D SpeedRange = Pair(Rand, TEXT("speed_scale_range"), FVector2D(1, 1));
    const auto Env = Object(ScenarioConfig, TEXT("environment"));
    AActor* Ground = SpawnSceneActor(TEXT("PTZ_Ground"));
    AddPart(Ground, TEXT("Ground"), CubeMesh, FVector(6000, 0, -25), FVector(400, 300, 0.5), Color(Env, TEXT("ground_color_rgb"), FLinearColor(0.26f, 0.30f, 0.26f)));
    const TArray<TSharedPtr<FJsonValue>>* JsonTargets = nullptr;
    ScenarioConfig->TryGetArrayField(TEXT("targets"), JsonTargets);
    TArray<TSharedPtr<FJsonValue>> ResolvedTargets;
    int Index = 0;
    for (const auto& V : *JsonTargets)
    {
        const auto T = V->AsObject();
        FPTZTargetRuntime Target;
        Target.Id = String(T, TEXT("id"), FString::Printf(TEXT("car_%d"), Index));
        Target.Path = Object(T, TEXT("path"));
        Target.Origin = Vector(Target.Path, TEXT("origin_cm"), FVector(1600, 0, 8));
        Target.Origin += FVector(Random.FRandRange(-Jitter, Jitter), Random.FRandRange(-Jitter, Jitter), 0);
        Target.SpeedScale = Random.FRandRange(SpeedRange.X, SpeedRange.Y);
        Target.Phase = Number(Target.Path, TEXT("phase_s"), 0);
        bool bRandomPhase = false;
        Target.Path->TryGetBoolField(TEXT("random_phase"), bRandomPhase);
        if (bRandomPhase) Target.Phase = Random.FRandRange(0.0, Number(Target.Path, TEXT("period_s"), 20));
        AActor* Car = SpawnSceneActor(TEXT("PTZ_Target_") + Target.Id);
        Car->Tags.Add(TEXT("PTZTarget"));
        Car->Tags.Add(FName(Target.Id));
        Target.Actor = Car;
        const FLinearColor C = Color(T, TEXT("color_rgb"), FLinearColor(0.9f, 0.018f, 0.012f));
        const FString VehicleKind = String(T, TEXT("vehicle_kind"), TEXT("car"));
        if (VehicleKind == TEXT("box_truck"))
        {
            // Local +X is forward, matching the existing native path orientation.
            AddPart(Car, TEXT("TruckChassis"), CubeMesh, FVector(0, 0, 72), FVector(11.8, 2.6, 0.65), FLinearColor(0.06f, 0.07f, 0.08f));
            AddPart(Car, TEXT("CargoBox"), CubeMesh, FVector(-110, 0, 267), FVector(9.2, 2.8, 3.6), FLinearColor(0.78f, 0.79f, 0.73f));
            AddPart(Car, TEXT("TruckCab"), CubeMesh, FVector(470, 0, 165), FVector(2.35, 2.6, 2.35), C);
            AddPart(Car, TEXT("Windshield"), CubeMesh, FVector(589, 0, 218), FVector(0.025, 2.3, 0.85), FLinearColor(0.035f, 0.06f, 0.08f));
            for (int Side : {-1, 1})
            {
                AddPart(Car, *FString::Printf(TEXT("CabWindow_%d"), Side), CubeMesh, FVector(468, Side * 131, 220), FVector(1.7, 0.025, 0.8), FLinearColor(0.035f, 0.06f, 0.08f));
                AddPart(Car, *FString::Printf(TEXT("CargoStripe_%d"), Side), CubeMesh, FVector(-110, Side * 141, 140), FVector(9.1, 0.025, 0.16), C);
                for (int Axle : {-420, -290, 440})
                    Target.Wheels.Add(AddPart(Car, *FString::Printf(TEXT("TruckWheel_%d_%d"), Axle, Side), SphereMesh, FVector(Axle, Side * 133, 36), FVector(0.65, 0.32, 0.65), FLinearColor(0.008f, 0.008f, 0.008f)));
            }
        }
        else
        {
            AddPart(Car, TEXT("CarBody"), CubeMesh, FVector(0, 0, 75), FVector(4, 1.9, 0.7), C);
            AddPart(Car, TEXT("CarRoof"), CubeMesh, FVector(-20, 0, 134), FVector(1.9, 1.55, 0.65), FLinearColor(0.035f, 0.06f, 0.08f));
            for (int X : {-1, 1}) for (int Y : {-1, 1})
                Target.Wheels.Add(AddPart(Car, *FString::Printf(TEXT("Wheel_%d_%d"), X, Y), SphereMesh, FVector(X * 125, Y * 99, 36), FVector(0.65, 0.30, 0.65), FLinearColor(0.008f, 0.008f, 0.008f)));
        }
        const double Scale = FMath::Clamp(Number(T, TEXT("scale"), 1), 0.1, 10.0);
        Car->SetActorScale3D(FVector(Scale));
        Targets.Add(Target);
        auto Resolved = MakeShared<FJsonObject>();
        Resolved->SetStringField(TEXT("id"), Target.Id);
        Resolved->SetStringField(TEXT("actor_name"), Car->GetName());
        Resolved->SetArrayField(TEXT("origin_cm"), Numbers({Target.Origin.X, Target.Origin.Y, Target.Origin.Z}));
        Resolved->SetNumberField(TEXT("speed_scale"), Target.SpeedScale);
        Resolved->SetNumberField(TEXT("phase_s"), Target.Phase);
        Resolved->SetNumberField(TEXT("scale"), Scale);
        Resolved->SetStringField(TEXT("vehicle_kind"), VehicleKind);
        ResolvedTargets.Add(MakeShared<FJsonValueObject>(Resolved));
        ++Index;
    }
    ResolvedConfig->SetArrayField(TEXT("targets"), ResolvedTargets);
    const TArray<TSharedPtr<FJsonValue>>* JsonOccluders = nullptr;
    if (ScenarioConfig->TryGetArrayField(TEXT("occluders"), JsonOccluders))
    {
        Index = 0;
        for (const auto& V : *JsonOccluders)
        {
            auto O = V->AsObject();
            const FString Kind = String(O, TEXT("kind"), TEXT("wall"));
            AActor* A = SpawnSceneActor(FString::Printf(TEXT("PTZ_Occluder_%d"), Index++));
            const FVector Size = Vector(O, TEXT("size_cm"), FVector(100, 500, 600));
            const FVector R = Vector(O, TEXT("rotation_deg"), FVector::ZeroVector);
            A->SetActorLocationAndRotation(Vector(O, TEXT("location_cm"), FVector(0, 0, 300)), FRotator(R.X, R.Y, R.Z));
            const FLinearColor C = Color(O, TEXT("color_rgb"), FLinearColor(0.4f, 0.42f, 0.44f));
            if (Kind == TEXT("tree"))
            {
                AddPart(A, TEXT("Trunk"), CubeMesh, FVector(0, 0, -Size.Z * 0.25), FVector(Size.X * 0.003, Size.Y * 0.003, Size.Z * 0.005), FLinearColor(0.15f, 0.08f, 0.03f));
                AddPart(A, TEXT("Crown"), SphereMesh, FVector(0, 0, Size.Z * 0.16), Size / 100, FLinearColor(0.04f, 0.18f, 0.035f));
            }
            else AddPart(A, TEXT("OpaqueGeometry"), Kind == TEXT("pillar") ? SphereMesh.Get() : CubeMesh.Get(), FVector::ZeroVector, Size / 100, C);
        }
    }
    ApplyEnvironment();
}

void APTZArena::ApplyEnvironment()
{
    const auto E = Object(ScenarioConfig, TEXT("environment"));
    for (TActorIterator<ADirectionalLight> It(GetWorld()); It; ++It)
    {
        auto* C = Cast<UDirectionalLightComponent>(It->GetLightComponent());
        C->SetMobility(EComponentMobility::Movable);
        C->SetIntensity(Number(E, TEXT("directional_lux"), 4));
    }
    for (TActorIterator<ASkyLight> It(GetWorld()); It; ++It)
    {
        It->GetLightComponent()->SetMobility(EComponentMobility::Movable);
        It->GetLightComponent()->SetIntensity(Number(E, TEXT("sky_intensity"), 0.8));
    }
    FActorSpawnParameters Params;
    Params.Name = TEXT("PTZ_Atmosphere");
    Params.NameMode = FActorSpawnParameters::ESpawnActorNameMode::Requested;
    auto* Fog = GetWorld()->SpawnActor<AExponentialHeightFog>(Params);
    Fog->GetComponent()->SetFogDensity(Number(E, TEXT("fog_density"), 0));
    Fog->GetComponent()->SetFogInscatteringColor(Color(E, TEXT("fog_color_rgb"), FLinearColor(0.62f, 0.68f, 0.72f)));
    SceneActors.Add(Fog);
    // Fixed exposure keeps lighting difficulty effective in RGB and the game viewport.
    const double Bias = FMath::Log2(FMath::Max(0.01, Number(E, TEXT("exposure"), 1)));
    Sensor()->SetAutoExposureBrightness(1, 1);
    Sensor()->SetExposureBias(Bias);
    auto& PP = ViewCamera->GetCameraComponent()->PostProcessSettings;
    PP.bOverride_AutoExposureMinBrightness = true;
    PP.bOverride_AutoExposureMaxBrightness = true;
    PP.AutoExposureMinBrightness = PP.AutoExposureMaxBrightness = 1;
    PP.bOverride_AutoExposureBias = true;
    PP.AutoExposureBias = Bias;
}

FVector APTZArena::TargetPosition(const FPTZTargetRuntime& T, double Time) const
{
    const FString Type = String(T.Path, TEXT("type"), TEXT("static"));
    const double Progress = IntegratedTime(T.Path, Time + T.Phase) * T.SpeedScale;
    if (Type == TEXT("static")) return T.Origin;
    if (Type == TEXT("line")) return T.Origin + Vector(T.Path, TEXT("velocity_cm_s"), FVector(0, 300, 0)) * Progress;
    const double Period = FMath::Max(0.1, Number(T.Path, TEXT("period_s"), 20));
    const double A = 2 * PI * Progress / Period;
    const FVector E = Vector(T.Path, TEXT("extent_cm"), FVector(700, 1100, 0));
    if (Type == TEXT("ellipse")) return T.Origin + FVector(E.X * FMath::Cos(A), E.Y * FMath::Sin(A), E.Z * FMath::Sin(A));
    if (Type == TEXT("figure8")) return T.Origin + FVector(E.X * FMath::Sin(A), E.Y * FMath::Sin(2 * A), E.Z * FMath::Sin(A));
    const TArray<TSharedPtr<FJsonValue>>* Points = nullptr;
    if (Type == TEXT("waypoints") && T.Path->TryGetArrayField(TEXT("waypoints_cm"), Points) && Points->Num() >= 2)
    {
        TArray<FVector> P;
        for (const auto& Value : *Points)
        {
            const auto& Row = Value->AsArray();
            if (Row.Num() == 3) P.Add(FVector(Row[0]->AsNumber(), Row[1]->AsNumber(), Row[2]->AsNumber()));
        }
        if (P.Num() < 2) return T.Origin;
        double Length = 0;
        for (int I = 0; I < P.Num(); ++I) Length += FVector::Dist(P[I], P[(I + 1) % P.Num()]);
        if (Length < 1) return P[0];
        double D = FMath::Fmod(Progress * Number(T.Path, TEXT("speed_cm_s"), 400), Length);
        if (D < 0) D += Length;
        for (int I = 0; I < P.Num(); ++I)
        {
            const FVector Next = P[(I + 1) % P.Num()];
            const double Segment = FVector::Dist(P[I], Next);
            if (D <= Segment && Segment > 0) return FMath::Lerp(P[I], Next, D / Segment) + (T.Origin - Vector(T.Path, TEXT("origin_cm"), FVector(1600, 0, 8)));
            D -= Segment;
        }
        return P[0];
    }
    return T.Origin;
}

void APTZArena::MoveVehicles(float VisualDeltaSeconds)
{
    for (auto& T : Targets) if (T.Actor.IsValid())
    {
        FVector P = TargetPosition(T, SimTime);
        const FVector Direction = TargetPosition(T, SimTime + 0.01) - P;
        FRotator R = Direction.SizeSquared() > 0.001 ? Direction.Rotation() : T.Actor->GetActorRotation();
        if (bRealtimeControl)
        {
            // The arena is a flat z=0 ground. Wheel center=36, radius=32.5 cm.
            // This visual-only mode grounds the tires; synchronous science is unchanged.
            P.Z = -3.5 * T.Actor->GetActorScale3D().Z;
            if (VisualDeltaSeconds > 0)
                R = FMath::RInterpConstantTo(T.Actor->GetActorRotation(), R, VisualDeltaSeconds, 120.f);
            if (T.bVisualPositionValid)
            {
                const double Distance = FVector::Dist2D(P, T.LastVisualPosition);
                T.WheelAngle = FMath::Fmod(T.WheelAngle + FMath::RadiansToDegrees(Distance / (32.5 * T.Actor->GetActorScale3D().X)), 360.0);
                for (auto& Wheel : T.Wheels) if (Wheel.IsValid()) Wheel->SetRelativeRotation(FRotator(-T.WheelAngle, 0, 0));
            }
            T.LastVisualPosition = P;
            T.bVisualPositionValid = true;
        }
        T.Actor->SetActorLocationAndRotation(P, R);
    }
}

void APTZArena::UpdateCamera()
{
    if (!Sensor() || !ViewCamera) return;
    Sensor()->SetSensorLocation(CameraLocation);
    Sensor()->SetSensorRotation(CameraRotation);
    Sensor()->SetSensorFOV(HorizontalFOV);
    ViewCamera->SetActorLocationAndRotation(CameraLocation, CameraRotation);
    ViewCamera->GetCameraComponent()->SetFieldOfView(HorizontalFOV);
    ViewCamera->GetCameraComponent()->SetAspectRatio(960.0f / 540.0f);
}

FString APTZArena::ResetEpisode(int32 Seed)
{
    if (AsyncRate > 0 || AsyncPending.Num()) SetAsyncCaptureRate(0);
    EpisodeSeed = Seed;
    SimTime = 0;
    bRealtimeControl = false;
    RealtimeActionRemaining = RealtimeElapsed = 0;
    RealtimeFrameIntervals.Reset();
    bDemo = bFreePreview = false;
    bDetectionReceived = bVisualDetectionVisible = false;
    ControlMode = TEXT("SYNCHRONOUS CONTROL - WAITING FOR PYTHON");
    FRandomStream Random(Seed);
    const auto C = Object(ScenarioConfig, TEXT("camera"));
    CameraLocation = Vector(C, TEXT("location_cm"), FVector(-2200, 0, 900));
    const FVector R = Vector(C, TEXT("rotation_deg"), FVector(-12, 0, 0)), J = Vector(C, TEXT("rotation_jitter_deg"), FVector::ZeroVector);
    CameraRotation = FRotator(R.X + Random.FRandRange(-J.X, J.X), R.Y + Random.FRandRange(-J.Y, J.Y), R.Z);
    PanSpeed = FMath::Clamp(Number(C, TEXT("pan_deg_s"), 25), 0.1, 180.0);
    TiltSpeed = FMath::Clamp(Number(C, TEXT("tilt_deg_s"), 25), 0.1, 180.0);
    ZoomFOVSpeed = FMath::Clamp(Number(C, TEXT("zoom_fov_deg_s"), 5), 0.1, 100.0);
    MinFOV = FMath::Clamp(Number(C, TEXT("fov_min_deg"), 4.4085), 1.0, 170.0);
    MaxFOV = FMath::Clamp(Number(C, TEXT("fov_max_deg"), 60), double(MinFOV), 170.0);
    HorizontalFOV = FMath::Clamp(Number(C, TEXT("fov_deg"), 60), double(MinFOV), double(MaxFOV));
    YawLimits = Pair(C, TEXT("yaw_limits_deg"), FVector2D(-40, 40));
    PitchLimits = Pair(C, TEXT("pitch_limits_deg"), FVector2D(-25, 25));
    bLegacyTargetCenterReset = false;
    C->TryGetBoolField(TEXT("legacy_target_center_reset"), bLegacyTargetCenterReset);
    BuildScenario(Random);
    MoveVehicles();
    if (bLegacyTargetCenterReset && Targets.Num() && Targets[0].Actor.IsValid())
    {
        CameraRotation = (Targets[0].Actor->GetActorLocation() + FVector(0, 0, 90) - CameraLocation).Rotation();
        CameraRotation.Yaw += Random.FRandRange(-10, 10);
        CameraRotation.Pitch += Random.FRandRange(-4, 4);
    }
    CameraRotation.Yaw = FMath::Clamp(CameraRotation.Yaw, YawLimits.X, YawLimits.Y);
    CameraRotation.Pitch = FMath::Clamp(CameraRotation.Pitch, PitchLimits.X, PitchLimits.Y);
    ResolvedConfig->SetArrayField(TEXT("initial_rotation_deg"), Numbers({CameraRotation.Pitch, CameraRotation.Yaw, CameraRotation.Roll}));
    auto ResolvedCamera = MakeShared<FJsonObject>();
    ResolvedCamera->SetArrayField(TEXT("location_cm"), Numbers({CameraLocation.X, CameraLocation.Y, CameraLocation.Z}));
    ResolvedCamera->SetArrayField(TEXT("yaw_limits_deg"), Numbers({YawLimits.X, YawLimits.Y}));
    ResolvedCamera->SetArrayField(TEXT("pitch_limits_deg"), Numbers({PitchLimits.X, PitchLimits.Y}));
    ResolvedCamera->SetNumberField(TEXT("pan_deg_s"), PanSpeed);
    ResolvedCamera->SetNumberField(TEXT("tilt_deg_s"), TiltSpeed);
    ResolvedCamera->SetNumberField(TEXT("zoom_fov_deg_s"), ZoomFOVSpeed);
    ResolvedCamera->SetNumberField(TEXT("fov_min_deg"), MinFOV);
    ResolvedCamera->SetNumberField(TEXT("fov_max_deg"), MaxFOV);
    ResolvedCamera->SetNumberField(TEXT("initial_fov_deg"), HorizontalFOV);
    ResolvedCamera->SetBoolField(TEXT("legacy_target_center_reset"), bLegacyTargetCenterReset);
    ResolvedConfig->SetObjectField(TEXT("camera"), ResolvedCamera);
    UpdateCamera();
    return GetState();
}

FString APTZArena::StepControl(float Pan, float Tilt, float Zoom, float Duration)
{
    bRealtimeControl = false;
    RealtimeActionRemaining = 0;
    bDetectionReceived = false;
    if (bDemo || bFreePreview) ControlMode = TEXT("SYNCHRONOUS CONTROL - WAITING FOR PYTHON");
    bDemo = bFreePreview = false;
    Duration = FMath::Clamp(Duration, 0.001f, 1.0f); // Declared precision-duration extension in isolated speed revision.
    CameraRotation.Yaw = FMath::Clamp(CameraRotation.Yaw + FMath::Clamp(Pan, -1.0f, 1.0f) * PanSpeed * Duration, YawLimits.X, YawLimits.Y);
    CameraRotation.Pitch = FMath::Clamp(CameraRotation.Pitch + FMath::Clamp(Tilt, -1.0f, 1.0f) * TiltSpeed * Duration, PitchLimits.X, PitchLimits.Y);
    HorizontalFOV = FMath::Clamp(HorizontalFOV - FMath::Clamp(Zoom, -1.0f, 1.0f) * ZoomFOVSpeed * Duration, MinFOV, MaxFOV);
    SimTime += Duration;
    MoveVehicles();
    UpdateCamera();
    return GetState();
}

void APTZArena::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);
    if (bRealtimeControl && Sensor())
    {
        const double Dt = FMath::Max(0.f, DeltaSeconds);
        const double ControlDt = FMath::Min(Dt, RealtimeActionRemaining);
        RealtimeElapsed += Dt;
        if (RealtimeFrameIntervals.Num() < 20000) RealtimeFrameIntervals.Add(Dt);
        // The scene keeps running during image transfer and Python inference.
        SimTime += Dt;
        CameraRotation.Yaw = FMath::Clamp(CameraRotation.Yaw + RealtimePan * PanSpeed * ControlDt, YawLimits.X, YawLimits.Y);
        CameraRotation.Pitch = FMath::Clamp(CameraRotation.Pitch + RealtimeTilt * TiltSpeed * ControlDt, PitchLimits.X, PitchLimits.Y);
        HorizontalFOV = FMath::Clamp(double(HorizontalFOV) - RealtimeZoom * ZoomFOVSpeed * ControlDt, double(MinFOV), double(MaxFOV));
        RealtimeActionRemaining = FMath::Max(0.0, RealtimeActionRemaining - ControlDt);
        MoveVehicles(DeltaSeconds);
        UpdateCamera();
        if (AsyncRate > 0) PumpAsyncFrames();
        return;
    }
    if (!(bDemo || bFreePreview) || !Sensor()) return;
    SimTime += FMath::Min(DeltaSeconds, 0.1f);
    MoveVehicles();
    if (bDemo && Targets.Num() && Targets[0].Actor.IsValid())
    {
        CameraRotation = (Targets[0].Actor->GetActorLocation() + FVector(0, 0, 90) - CameraLocation).Rotation();
        HorizontalFOV = 38;
        UpdateCamera();
    }
}

void APTZArena::SetRealtime(bool Enabled)
{
    if (Enabled && !bRealtimeControl)
    {
        RealtimeElapsed = 0;
        RealtimeFrameIntervals.Reset();
        for (auto& T : Targets)
        {
            T.bVisualPositionValid = false;
            // Visible spokes make wheel rotation observable on the geometric tires.
            for (auto& Wheel : T.Wheels) if (Wheel.IsValid() && Wheel->GetNumChildrenComponents() == 0)
            {
                for (int Side : {-1, 1})
                {
                    auto* Spoke = AddPart(T.Actor.Get(), *FString::Printf(TEXT("Spoke_%s_%d"), *Wheel->GetName(), Side), CubeMesh,
                        FVector(0, Side * 51, 0), FVector(0.72, 0.035, 0.10), FLinearColor(0.65f, 0.65f, 0.65f));
                    Spoke->AttachToComponent(Wheel.Get(), FAttachmentTransformRules::KeepRelativeTransform);
                }
            }
        }
    }
    bRealtimeControl = Enabled;
    bDemo = bFreePreview = false;
    RealtimeActionRemaining = 0;
    RealtimePan = RealtimeTilt = RealtimeZoom = 0;
    bDetectionReceived = bVisualDetectionVisible = false;
    MoveVehicles();
}

FString APTZArena::PlayControl(float Pan, float Tilt, float Zoom, float Duration, bool CompactReply)
{
    if (!bRealtimeControl) SetRealtime(true);
    RealtimePan = FMath::Clamp(Pan, -1.f, 1.f);
    RealtimeTilt = FMath::Clamp(Tilt, -1.f, 1.f);
    RealtimeZoom = FMath::Clamp(Zoom, -1.f, 1.f);
    RealtimeActionRemaining = FMath::Clamp(Duration, 0.001f, 1.f);
    bDetectionReceived = false;
    if (CompactReply)
    {
        auto Reply = MakeShared<FJsonObject>();
        Reply->SetNumberField(TEXT("sim_time"), SimTime);
        Reply->SetNumberField(TEXT("action_remaining_seconds"), RealtimeActionRemaining);
        Reply->SetArrayField(TEXT("rotation"), Numbers({CameraRotation.Pitch, CameraRotation.Yaw, CameraRotation.Roll}));
        Reply->SetNumberField(TEXT("fov"), HorizontalFOV);
        FString Text; FJsonSerializer::Serialize(Reply, TJsonWriterFactory<>::Create(&Text)); return Text;
    }
    return GetState();
}

void APTZArena::EndPlay(const EEndPlayReason::Type Reason)
{
    if (AsyncRate > 0 || AsyncPending.Num()) SetAsyncCaptureRate(0);
    if (AsyncStreamRegion)
    {
        FPlatformMemory::UnmapNamedSharedMemoryRegion(AsyncStreamRegion);
        AsyncStreamRegion = nullptr;
    }
    if (SharedFrameRegion)
    {
        FPlatformMemory::UnmapNamedSharedMemoryRegion(SharedFrameRegion);
        SharedFrameRegion = nullptr;
    }
    Super::EndPlay(Reason);
}

FString APTZArena::SetAsyncCaptureRate(float Rate)
{
    if (AsyncRate > 0 || AsyncPending.Num())
    {
        AsyncRate = 0;
        FlushRenderingCommands(); // Only shutdown/reconfiguration, never per frame.
        AsyncPending.Empty(); AsyncFree.Empty();
    }
    if (AsyncDisabledBackgroundViewport && GetWorld() && GetWorld()->GetGameViewport())
    {
        GetWorld()->GetGameViewport()->bDisableWorldRendering = false;
        AsyncDisabledBackgroundViewport = false;
    }
    AsyncLatestPacket.Reset(); AsyncReadySequence = AsyncIssuedSequence = 0;
    if (Rate <= 0) return GetState();
    if (!bRealtimeControl || !Sensor()) return TEXT("{\"error\":\"Start real-time scene first\"}");
    TArray<USceneComponent*> SensorChildren;
    Sensor()->GetChildrenComponents(true, SensorChildren);
    for (USceneComponent* C : SensorChildren) if (auto* Lit = Cast<ULitCamSensor>(C)) { AsyncLitSensor = Lit; break; }
    if (!AsyncLitSensor.IsValid() || !AsyncLitSensor->TextureTarget) return TEXT("{\"error\":\"Lit target unavailable\"}");
    if (!AsyncStreamRegion)
    {
        AsyncStreamName = FGuid::NewGuid().ToString(EGuidFormats::DigitsWithHyphensInBraces);
        AsyncStreamRegion = FPlatformMemory::MapNamedSharedMemoryRegion(AsyncStreamName, true,
            FPlatformMemory::ESharedMemoryAccess::Read | FPlatformMemory::ESharedMemoryAccess::Write, 64 + 4 * 4 * 1024 * 1024);
    }
    if (!AsyncStreamRegion) return TEXT("{\"error\":\"Stream mapping unavailable\"}");
    FMemory::Memzero(AsyncStreamRegion->GetAddress(), 64);
    // Headless recordings use the identical Lit scene capture. Rendering an
    // additional unused game viewport duplicates GPU work without video data.
    if (GetWorld()->GetGameViewport() && !GetWorld()->GetGameViewport()->bDisableWorldRendering)
    {
        GetWorld()->GetGameViewport()->bDisableWorldRendering = true;
        AsyncDisabledBackgroundViewport = true;
    }
    AsyncRate = FMath::Clamp(double(Rate), 1., 120.);
    AsyncNextCapture = FPlatformTime::Seconds();
    TSharedPtr<FJsonObject> Reply;
    FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(GetState()), Reply);
    Reply->SetStringField(TEXT("stream_shared_name"), AsyncStreamName);
    Reply->SetNumberField(TEXT("stream_capacity_bytes"), 64 + 4 * 4 * 1024 * 1024);
    Reply->SetNumberField(TEXT("native_platform_seconds"), FPlatformTime::Seconds());
    FString ReplyText; FJsonSerializer::Serialize(Reply.ToSharedRef(), TJsonWriterFactory<>::Create(&ReplyText)); return ReplyText;
}

void APTZArena::PumpAsyncFrames()
{
    // Drain completed GPU copies without waiting for GPU or flushing rendering.
    while (AsyncPending.Num())
    {
        auto Frame = AsyncPending[0];
        if (!Frame->Submitted.Load() || !Frame->Readback->IsReady()) break;
        int32 RowPitch = 0, BufferHeight = 0;
        uint8* Data = static_cast<uint8*>(Frame->Readback->Lock(RowPitch, &BufferHeight));
        TArray<FColor> Pixels; Pixels.SetNumUninitialized(960 * 540);
        FReadSurfaceDataFlags Flags; Flags.SetLinearToGamma(false);
        const bool Valid = Data && RowPitch >= 960 && BufferHeight >= 540 && ConvertRAWSurfaceDataToFColor(
            Frame->Format, 960, 540, Data, RowPitch * GPixelFormats[Frame->Format].BlockBytes, Pixels.GetData(), Flags);
        Frame->Readback->Unlock();
        if (Valid)
        {
            Frame->Metadata->SetNumberField(TEXT("capture_readback_seconds"), FPlatformTime::Seconds() - Frame->CaptureTime);
            Frame->Metadata->SetNumberField(TEXT("async_source_sequence"), Frame->Sequence);
            Frame->Metadata->SetNumberField(TEXT("async_capture_platform_seconds"), Frame->CaptureTime);
            Frame->Metadata->SetStringField(TEXT("capture_backend"), TEXT("GPU_async_same_scene_metadata_original_pixel_conversion"));
            FString Text; FJsonSerializer::Serialize(Frame->Metadata.ToSharedRef(), TJsonWriterFactory<>::Create(&Text));
            const FTCHARToUTF8 Json(*Text);
            const uint32 Header[] = {0x315A5450, uint32(Json.Length()), 960, 540};
            AsyncLatestPacket.Reset();
            AsyncLatestPacket.Append(reinterpret_cast<const uint8*>(Header), sizeof(Header));
            AsyncLatestPacket.Append(reinterpret_cast<const uint8*>(Json.Get()), Json.Length());
            AsyncLatestPacket.Append(reinterpret_cast<const uint8*>(Pixels.GetData()), Pixels.Num() * sizeof(FColor));
            AsyncReadySequence = Frame->Sequence; AsyncLatestCaptureTime = Frame->CaptureTime;
            if (AsyncStreamRegion && AsyncLatestPacket.Num() <= 4 * 1024 * 1024)
            {
                constexpr uint32 SlotCapacity = 4 * 1024 * 1024;
                uint8* Buffer = static_cast<uint8*>(AsyncStreamRegion->GetAddress());
                volatile int64* Version = reinterpret_cast<volatile int64*>(Buffer);
                const uint32 Slot = uint32(Frame->Sequence % 4), Size = AsyncLatestPacket.Num();
                // Version is odd while publishing. Reader validates the same
                // even version before/after its private copy. No torn images.
                FPlatformAtomics::InterlockedExchange(Version, int64(Frame->Sequence * 2 + 1));
                FMemory::Memcpy(Buffer + 64 + Slot * SlotCapacity, AsyncLatestPacket.GetData(), Size);
                *reinterpret_cast<uint64*>(Buffer + 8) = Frame->Sequence;
                *reinterpret_cast<uint32*>(Buffer + 16) = Size;
                *reinterpret_cast<uint32*>(Buffer + 20) = Slot;
                *reinterpret_cast<uint32*>(Buffer + 24) = SlotCapacity;
                *reinterpret_cast<uint32*>(Buffer + 28) = 0x325A5450;
                *reinterpret_cast<uint32*>(Buffer + 32) = 4;
                FPlatformAtomics::InterlockedExchange(Version, int64(Frame->Sequence * 2));
            }
        }
        AsyncPending.RemoveAt(0);
        Frame->Submitted.Store(false);
        AsyncFree.Add(Frame);
    }
    const double Now = FPlatformTime::Seconds();
    if (Now < AsyncNextCapture || AsyncPending.Num() >= 4 || !AsyncLitSensor.IsValid()) return;
    AsyncNextCapture += 1. / AsyncRate;
    if (Now - AsyncNextCapture > 1. / AsyncRate) AsyncNextCapture = Now + 1. / AsyncRate;
    auto Frame = AsyncFree.Num() ? AsyncFree.Pop() : MakeShared<FPTZPendingGPUFrame, ESPMode::ThreadSafe>();
    Frame->Sequence = ++AsyncIssuedSequence; Frame->CaptureTime = Now;
    FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(GetState()), Frame->Metadata);
    auto* Lit = AsyncLitSensor.Get();
    Frame->Format = Lit->TextureTarget->GetFormat();
    if (!Frame->Readback) Frame->Readback = MakeUnique<FRHIGPUTextureReadback>(TEXT("PTZ_async_RGB"));
    Lit->UpdateCineCameraView();
    Lit->CaptureScene(); // Pushes scene transforms, queues rendering, no ReadPixels.
    FTextureRenderTargetResource* Resource = Lit->TextureTarget->GameThread_GetRenderTargetResource();
    ENQUEUE_RENDER_COMMAND(PTZCopyRGB)([Frame, Resource](FRHICommandListImmediate& RHICmdList)
    {
        FRHITexture* Texture = Resource->GetRenderTargetTexture();
        RHICmdList.Transition(FRHITransitionInfo(Texture, ERHIAccess::SRVMask, ERHIAccess::CopySrc));
        Frame->Readback->EnqueueCopy(RHICmdList, Texture);
        RHICmdList.Transition(FRHITransitionInfo(Texture, ERHIAccess::CopySrc, ERHIAccess::SRVMask));
        Frame->Submitted.Store(true);
    });
    AsyncPending.Add(Frame);
}

FString APTZArena::CaptureAsyncSharedFrame()
{
    if (!AsyncLatestPacket.Num()) return TEXT("{\"ready\":false}");
    if (!SharedFrameRegion)
    {
        SharedFrameName = FGuid::NewGuid().ToString(EGuidFormats::DigitsWithHyphensInBraces);
        SharedFrameRegion = FPlatformMemory::MapNamedSharedMemoryRegion(SharedFrameName, true,
            FPlatformMemory::ESharedMemoryAccess::Read | FPlatformMemory::ESharedMemoryAccess::Write, 4 * 1024 * 1024);
    }
    if (!SharedFrameRegion) return TEXT("{\"error\":\"Shared mapping unavailable\"}");
    FMemory::Memcpy(SharedFrameRegion->GetAddress(), AsyncLatestPacket.GetData(), AsyncLatestPacket.Num());
    auto Reply = MakeShared<FJsonObject>();
    Reply->SetBoolField(TEXT("ready"), true);
    Reply->SetStringField(TEXT("shared_name"), SharedFrameName);
    Reply->SetNumberField(TEXT("packet_bytes"), AsyncLatestPacket.Num());
    Reply->SetNumberField(TEXT("capacity_bytes"), 4 * 1024 * 1024);
    Reply->SetNumberField(TEXT("sequence"), AsyncReadySequence);
    Reply->SetNumberField(TEXT("sample_age_seconds"), FPlatformTime::Seconds() - AsyncLatestCaptureTime);
    FString Text; FJsonSerializer::Serialize(Reply, TJsonWriterFactory<>::Create(&Text)); return Text;
}

FString APTZArena::CaptureSharedFrame()
{
    constexpr SIZE_T Capacity = 4 * 1024 * 1024;
    if (!SharedFrameRegion)
    {
        // A brace-form GUID maps in the current session without global/admin
        // privileges on Windows. It is unique to this arena's lifetime.
        SharedFrameName = FGuid::NewGuid().ToString(EGuidFormats::DigitsWithHyphensInBraces);
        SharedFrameRegion = FPlatformMemory::MapNamedSharedMemoryRegion(SharedFrameName, true,
            FPlatformMemory::ESharedMemoryAccess::Read | FPlatformMemory::ESharedMemoryAccess::Write, Capacity);
    }
    if (!SharedFrameRegion) return TEXT("{\"error\":\"Shared image mapping unavailable\"}");
    const TArray<uint8> Packet = CaptureRealtimeFrame();
    if (Packet.Num() == 0 || Packet.Num() > Capacity) return TEXT("{\"error\":\"Invalid shared image packet\"}");
    FMemory::Memcpy(SharedFrameRegion->GetAddress(), Packet.GetData(), Packet.Num());
    // One client owns all IO: the next frame cannot overwrite this memory
    // until that client has copied this packet and issued its next request.
    auto Reply = MakeShared<FJsonObject>();
    Reply->SetStringField(TEXT("shared_name"), SharedFrameName);
    Reply->SetNumberField(TEXT("packet_bytes"), Packet.Num());
    Reply->SetNumberField(TEXT("capacity_bytes"), Capacity);
    Reply->SetNumberField(TEXT("sequence"), ++SharedFrameSequence);
    FString Text;
    FJsonSerializer::Serialize(Reply, TJsonWriterFactory<>::Create(&Text));
    return Text;
}

TArray<uint8> APTZArena::CaptureRealtimeFrame()
{
    // One game-thread command returns RGB and diagnostics from the same scene time.
    // Raw BGRA avoids PNG compression/decompression and a second state RPC.
    const double Start = FPlatformTime::Seconds();
    TArray<FColor> Pixels;
    int Width = 0, Height = 0;
    Sensor()->GetLit(Pixels, Width, Height);
    const double CaptureSeconds = FPlatformTime::Seconds() - Start;
    TSharedPtr<FJsonObject> Metadata;
    if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(GetState()), Metadata) || !Metadata.IsValid()
        || Width <= 0 || Height <= 0 || Pixels.Num() != Width * Height)
        return {};
    Metadata->SetNumberField(TEXT("capture_readback_seconds"), CaptureSeconds);
    FString MetadataText;
    FJsonSerializer::Serialize(Metadata.ToSharedRef(), TJsonWriterFactory<>::Create(&MetadataText));
    const FTCHARToUTF8 Json(*MetadataText);
    const uint32 Header[] = {0x315A5450, uint32(Json.Length()), uint32(Width), uint32(Height)};
    TArray<uint8> Packet;
    Packet.Append(reinterpret_cast<const uint8*>(Header), sizeof(Header));
    Packet.Append(reinterpret_cast<const uint8*>(Json.Get()), Json.Length());
    Packet.Append(reinterpret_cast<const uint8*>(Pixels.GetData()), Pixels.Num() * sizeof(FColor));
    return Packet;
}

FString APTZArena::GetRealtimeTiming() const
{
    auto S = MakeShared<FJsonObject>();
    TArray<TSharedPtr<FJsonValue>> Intervals;
    for (double Dt : RealtimeFrameIntervals) Intervals.Add(MakeShared<FJsonValueNumber>(Dt));
    S->SetArrayField(TEXT("motion_tick_intervals_seconds"), Intervals);
    S->SetNumberField(TEXT("motion_elapsed_seconds"), RealtimeElapsed);
    S->SetStringField(TEXT("scope"), TEXT("UE game ticks that advance cars and camera, not detector FPS or encoded video FPS"));
    FString Text;
    FJsonSerializer::Serialize(S, TJsonWriterFactory<>::Create(&Text));
    return Text;
}

void APTZArena::SetDemo(bool Enabled)
{
    bDemo = Enabled; bFreePreview = false; bDetectionReceived = false;
    ControlMode = Enabled ? TEXT("SIMULATOR GROUND-TRUTH FOLLOW PREVIEW") : TEXT("SYNCHRONOUS CONTROL - WAITING FOR PYTHON");
}
void APTZArena::SetPreview(bool Enabled)
{
    bFreePreview = Enabled; bDemo = false; bDetectionReceived = false;
    ControlMode = Enabled ? TEXT("SCENE MOTION PREVIEW - FIXED CAMERA") : TEXT("SYNCHRONOUS CONTROL - WAITING FOR PYTHON");
}
void APTZArena::SetMode(const FString& Mode) { ControlMode = Mode.Left(96); }
FString APTZArena::GetDisplayMode() const { return ControlMode + TEXT(" | ") + String(ScenarioConfig, TEXT("name"), TEXT("unknown")); }
FString APTZArena::SetVisualDetection(FVector4f Box, bool Visible)
{
    VisualDetectionBox = Box; bVisualDetectionVisible = Visible; bDetectionReceived = true;
    return GetState();
}

TSharedRef<FJsonObject> APTZArena::TargetState(const FPTZTargetRuntime& T) const
{
    auto S = MakeShared<FJsonObject>();
    S->SetStringField(TEXT("id"), T.Id);
    if (!T.Actor.IsValid()) return S;
    const FVector P = T.Actor->GetActorLocation();
    S->SetStringField(TEXT("actor_name"), T.Actor->GetName());
    S->SetArrayField(TEXT("location_cm"), Numbers({P.X, P.Y, P.Z}));
    FVector Origin, Extent;
    T.Actor->GetActorBounds(false, Origin, Extent);
    const double W = Sensor()->GetFilmWidth(), H = Sensor()->GetFilmHeight();
    const double Focal = W / (2 * FMath::Tan(FMath::DegreesToRadians(HorizontalFOV * 0.5)));
    FVector2D Low(DBL_MAX, DBL_MAX), High(-DBL_MAX, -DBL_MAX);
    bool InFront = true;
    for (int X : {-1, 1}) for (int Y : {-1, 1}) for (int Z : {-1, 1})
    {
        const FVector Local = CameraRotation.UnrotateVector(Origin + Extent * FVector(X, Y, Z) - CameraLocation);
        if (Local.X <= 1) { InFront = false; continue; }
        const FVector2D Pixel(W * 0.5 + Focal * Local.Y / Local.X, H * 0.5 - Focal * Local.Z / Local.X);
        Low.X = FMath::Min(Low.X, Pixel.X); Low.Y = FMath::Min(Low.Y, Pixel.Y);
        High.X = FMath::Max(High.X, Pixel.X); High.Y = FMath::Max(High.Y, Pixel.Y);
    }
    const bool InFrame = InFront && Low.X < W && High.X > 0 && Low.Y < H && High.Y > 0;
    S->SetBoolField(TEXT("gt_in_frame"), InFrame);
    if (InFront)
    {
        S->SetArrayField(TEXT("gt_bbox_normalized"), Numbers({Low.X / W, Low.Y / H, High.X / W, High.Y / H}));
        S->SetArrayField(TEXT("gt_bbox_clipped_normalized"), Numbers({FMath::Clamp(Low.X / W, 0.0, 1.0), FMath::Clamp(Low.Y / H, 0.0, 1.0), FMath::Clamp(High.X / W, 0.0, 1.0), FMath::Clamp(High.Y / H, 0.0, 1.0)}));
    }
    FHitResult Hit;
    FCollisionQueryParams Query(SCENE_QUERY_STAT(PTZVisibility), true);
    Query.AddIgnoredActor(CaptureCamera); Query.AddIgnoredActor(ViewCamera); Query.AddIgnoredActor(this);
    // Ignore invisible editor actors even if an older saved map retained collision.
    for (TActorIterator<AActor> It(GetWorld()); It; ++It)
        if (It->IsHidden() || It->Tags.Contains(TEXT("PTZ_EditorContext"))) Query.AddIgnoredActor(*It);
    const bool Blocked = GetWorld()->LineTraceSingleByChannel(Hit, CameraLocation, Origin, ECC_Visibility, Query);
    S->SetBoolField(TEXT("gt_center_occluded"), Blocked && Hit.GetActor() != T.Actor.Get());
    S->SetStringField(TEXT("gt_occluder"), Blocked && Hit.GetActor() != T.Actor.Get() && Hit.GetActor() ? Hit.GetActor()->GetName() : TEXT(""));
    return S;
}

FString APTZArena::GetState() const
{
    auto S = MakeShared<FJsonObject>();
    S->SetStringField(TEXT("scene"), TEXT("PTZ_Arena_v1"));
    S->SetStringField(TEXT("platform"), TEXT("PTZSim UE 5.7 replacement, not author CamSim"));
    int ArenaCount = 0;
    for (TActorIterator<APTZArena> It(GetWorld()); It; ++It) ++ArenaCount;
    S->SetNumberField(TEXT("active_arena_count"), ArenaCount);
    S->SetStringField(TEXT("scenario_name"), String(ScenarioConfig, TEXT("name"), TEXT("unknown")));
    if (ScenarioConfig) S->SetObjectField(TEXT("scenario_config"), ScenarioConfig);
    if (ResolvedConfig) S->SetObjectField(TEXT("resolved_config"), ResolvedConfig);
    S->SetNumberField(TEXT("seed"), EpisodeSeed); S->SetNumberField(TEXT("sim_time"), SimTime);
    S->SetBoolField(TEXT("demo_geometry_follow"), bDemo); S->SetBoolField(TEXT("free_preview"), bFreePreview);
    S->SetBoolField(TEXT("realtime_control"), bRealtimeControl);
    S->SetNumberField(TEXT("action_remaining_seconds"), RealtimeActionRemaining);
    S->SetNumberField(TEXT("motion_tick_count"), RealtimeFrameIntervals.Num());
    S->SetStringField(TEXT("control_mode"), ControlMode);
    S->SetBoolField(TEXT("legacy_target_center_reset"), bLegacyTargetCenterReset);
    S->SetBoolField(TEXT("image_detection_received"), bDetectionReceived);
    S->SetBoolField(TEXT("image_detection_visible"), bDetectionReceived && bVisualDetectionVisible);
    S->SetArrayField(TEXT("image_detection_bbox"), Numbers({VisualDetectionBox.X, VisualDetectionBox.Y, VisualDetectionBox.Z, VisualDetectionBox.W}));
    S->SetArrayField(TEXT("rotation"), Numbers({CameraRotation.Pitch, CameraRotation.Yaw, CameraRotation.Roll}));
    S->SetArrayField(TEXT("camera_location"), Numbers({CameraLocation.X, CameraLocation.Y, CameraLocation.Z}));
    S->SetNumberField(TEXT("fov"), HorizontalFOV);
    S->SetNumberField(TEXT("focal_ratio"), FMath::Tan(FMath::DegreesToRadians(MaxFOV * 0.5)) / FMath::Tan(FMath::DegreesToRadians(HorizontalFOV * 0.5)));
    S->SetStringField(TEXT("zoom_model"), TEXT("perspective FOV; focal_ratio is geometric equivalent, not physical optical simulation"));
    const TArray<UFusionCamSensor*> Sensors = USensorBPLib::GetFusionSensorList();
    S->SetNumberField(TEXT("camera_id"), Sensors.IndexOfByKey(Sensor()));
    TArray<TSharedPtr<FJsonValue>> JsonTargets;
    for (const auto& T : Targets) JsonTargets.Add(MakeShared<FJsonValueObject>(TargetState(T)));
    S->SetArrayField(TEXT("targets"), JsonTargets);
    if (JsonTargets.Num())
    {
        const auto T = JsonTargets[0]->AsObject();
        S->SetArrayField(TEXT("target_location"), T->GetArrayField(TEXT("location_cm")));
        S->SetBoolField(TEXT("gt_in_frame"), T->GetBoolField(TEXT("gt_in_frame")));
        if (T->HasField(TEXT("gt_bbox_normalized"))) S->SetArrayField(TEXT("gt_bbox_normalized"), T->GetArrayField(TEXT("gt_bbox_normalized")));
    }
    S->SetStringField(TEXT("gt_usage"), TEXT("diagnostics/training detector labels only; policy/reward cannot read targets, target_location, gt_bbox or resolved_config"));
    FString Text;
    FJsonSerializer::Serialize(S, TJsonWriterFactory<>::Create(&Text));
    return Text;
}

void APTZArena::RegisterCommands()
{
    const TWeakObjectPtr<APTZArena> Weak(this);
    const auto D = FUnrealcvServer::Get().GetCommandDispatcher();
    D->BindCommand(TEXT("vset /ptz/realtime [uint]"), FDispatcherDelegate::CreateLambda([Weak](const TArray<FString>& A)
    {
        if (!Weak.IsValid()) return FExecStatus::Error(TEXT("Start Play first"));
        Weak->SetRealtime(FCString::Atoi(*A[0]) != 0);
        return FExecStatus::OK(Weak->GetState());
    }), TEXT("Real-time image-policy demo only; cars advance during Python inference"));
    D->BindCommand(TEXT("vset /ptz/play [float] [float] [float] [float]"), FDispatcherDelegate::CreateLambda([Weak](const TArray<FString>& A)
    {
        if (!Weak.IsValid()) return FExecStatus::Error(TEXT("Start Play first"));
        for (const auto& V : A) if (!FMath::IsFinite(FCString::Atof(*V))) return FExecStatus::Error(TEXT("Actions must be finite"));
        return FExecStatus::OK(Weak->PlayControl(FCString::Atof(*A[0]), FCString::Atof(*A[1]), FCString::Atof(*A[2]), FCString::Atof(*A[3])));
    }), TEXT("Execute a timed camera action over actual UE frames; no instant pose jump"));
    D->BindCommand(TEXT("vget /ptz/frame"), FDispatcherDelegate::CreateLambda([Weak](const TArray<FString>&)
    { return Weak.IsValid() && Weak->Sensor() ? FExecStatus::Binary(Weak->CaptureRealtimeFrame()) : FExecStatus::Error(TEXT("Start Play first")); }), TEXT("Atomic PTZ1 header + JSON + raw BGRA capture, without PNG encoding"));
    D->BindCommand(TEXT("vset /ptz/play_fast [float] [float] [float] [float]"), FDispatcherDelegate::CreateLambda([Weak](const TArray<FString>& A)
    {
        if (!Weak.IsValid()) return FExecStatus::Error(TEXT("Start Play first"));
        for (const auto& V : A) if (!FMath::IsFinite(FCString::Atof(*V))) return FExecStatus::Error(TEXT("Actions must be finite"));
        return FExecStatus::OK(Weak->PlayControl(FCString::Atof(*A[0]), FCString::Atof(*A[1]), FCString::Atof(*A[2]), FCString::Atof(*A[3]), true));
    }), TEXT("Identical full-duration motor pulse; compact own-camera acknowledgement only"));
    D->BindCommand(TEXT("vget /ptz/frame_shm"), FDispatcherDelegate::CreateLambda([Weak](const TArray<FString>&)
    { return Weak.IsValid() && Weak->Sensor() ? FExecStatus::OK(Weak->CaptureSharedFrame()) : FExecStatus::Error(TEXT("Start Play first")); }), TEXT("Same PTZ1 RGB/metadata packet via local shared memory; no changed action or simulation semantics"));
    D->BindCommand(TEXT("vset /ptz/async_stream [float]"), FDispatcherDelegate::CreateLambda([Weak](const TArray<FString>& A)
    { return Weak.IsValid() ? FExecStatus::OK(Weak->SetAsyncCaptureRate(FCString::Atof(*A[0]))) : FExecStatus::Error(TEXT("Start Play first")); }), TEXT("Enable timestamped asynchronous GPU RGB capture at requested real-world cadence, zero disables"));
    D->BindCommand(TEXT("vget /ptz/frame_async"), FDispatcherDelegate::CreateLambda([Weak](const TArray<FString>&)
    { return Weak.IsValid() ? FExecStatus::OK(Weak->CaptureAsyncSharedFrame()) : FExecStatus::Error(TEXT("Start Play first")); }), TEXT("Copy newest completed GPU image/own-time metadata into shared packet; repeated source sequence is not a new frame"));
    D->BindCommand(TEXT("vget /ptz/timing"), FDispatcherDelegate::CreateLambda([Weak](const TArray<FString>&)
    { return Weak.IsValid() ? FExecStatus::OK(Weak->GetRealtimeTiming()) : FExecStatus::Error(TEXT("Start Play first")); }), TEXT("Actual UE motion frame intervals for latency verification"));
    D->BindCommand(TEXT("vget /ptz/state"), FDispatcherDelegate::CreateLambda([Weak](const TArray<FString>&)
    { return Weak.IsValid() ? FExecStatus::OK(Weak->GetState()) : FExecStatus::Error(TEXT("Start Play first")); }), TEXT("PTZ scene state; GT diagnostics only"));
    D->BindCommand(TEXT("vset /ptz/reset [uint]"), FDispatcherDelegate::CreateLambda([Weak](const TArray<FString>& A)
    { return Weak.IsValid() ? FExecStatus::OK(Weak->ResetEpisode(FCString::Atoi(*A[0]))) : FExecStatus::Error(TEXT("Start Play first")); }), TEXT("Seeded synchronous reset of current scenario"));
    D->BindCommand(TEXT("vset /ptz/scenario [str]"), FDispatcherDelegate::CreateLambda([Weak](const TArray<FString>& A)
    {
        if (!Weak.IsValid()) return FExecStatus::Error(TEXT("Start Play first"));
        FString Error;
        return Weak->LoadScenario(A[0], Error) ? FExecStatus::OK(Weak->GetState()) : FExecStatus::Error(Error);
    }), TEXT("Load JSON scenario path, then reset synchronously"));
    D->BindCommand(TEXT("vset /ptz/step [float] [float] [float] [float]"), FDispatcherDelegate::CreateLambda([Weak](const TArray<FString>& A)
    {
        if (!Weak.IsValid()) return FExecStatus::Error(TEXT("Start Play first"));
        for (const auto& V : A) if (!FMath::IsFinite(FCString::Atof(*V))) return FExecStatus::Error(TEXT("Actions must be finite"));
        return FExecStatus::OK(Weak->StepControl(FCString::Atof(*A[0]), FCString::Atof(*A[1]), FCString::Atof(*A[2]), FCString::Atof(*A[3])));
    }), TEXT("Synchronous rates [-1,1], duration [0.02,1]; paper policy [0.04,1]"));
    D->BindCommand(TEXT("vset /ptz/view [float] [float] [float]"), FDispatcherDelegate::CreateLambda([Weak](const TArray<FString>& A)
    {
        if (!Weak.IsValid()) return FExecStatus::Error(TEXT("Start Play first"));
        for (const auto& V : A) if (!FMath::IsFinite(FCString::Atof(*V))) return FExecStatus::Error(TEXT("View parameters must be finite"));
        Weak->bDemo = Weak->bFreePreview = false; Weak->bDetectionReceived = false;
        Weak->CameraRotation.Pitch = FMath::Clamp(double(FCString::Atof(*A[0])), Weak->PitchLimits.X, Weak->PitchLimits.Y);
        Weak->CameraRotation.Yaw = FMath::Clamp(double(FCString::Atof(*A[1])), Weak->YawLimits.X, Weak->YawLimits.Y);
        Weak->HorizontalFOV = FMath::Clamp(FCString::Atof(*A[2]), Weak->MinFOV, Weak->MaxFOV);
        Weak->ControlMode = TEXT("DATASET / DEBUG CAMERA VIEW");
        Weak->UpdateCamera(); return FExecStatus::OK(Weak->GetState());
    }), TEXT("Dataset/debug: absolute pitch yaw FOV, no target alignment or time advance"));
    D->BindCommand(TEXT("vset /ptz/demo [uint]"), FDispatcherDelegate::CreateLambda([Weak](const TArray<FString>& A)
    {
        if (!Weak.IsValid()) return FExecStatus::Error(TEXT("Start Play first"));
        Weak->SetDemo(FCString::Atoi(*A[0]) != 0); return FExecStatus::OK(Weak->GetState());
    }), TEXT("Explicit simulator-GT-follow preview, never an RL result"));
    D->BindCommand(TEXT("vset /ptz/preview [uint]"), FDispatcherDelegate::CreateLambda([Weak](const TArray<FString>& A)
    {
        if (!Weak.IsValid()) return FExecStatus::Error(TEXT("Start Play first"));
        Weak->SetPreview(FCString::Atoi(*A[0]) != 0); return FExecStatus::OK(Weak->GetState());
    }), TEXT("Scene motion preview with a fixed camera"));
    D->BindCommand(TEXT("vset /ptz/mode [str]"), FDispatcherDelegate::CreateLambda([Weak](const TArray<FString>& A)
    {
        if (!Weak.IsValid()) return FExecStatus::Error(TEXT("Start Play first"));
        Weak->SetMode(A[0]); return FExecStatus::OK(Weak->GetState());
    }), TEXT("HUD policy source label, e.g. HAS-CAM, HSV-P, PPO-baseline"));
    D->BindCommand(TEXT("vset /ptz/detection [float] [float] [float] [float] [uint]"), FDispatcherDelegate::CreateLambda([Weak](const TArray<FString>& A)
    {
        if (!Weak.IsValid()) return FExecStatus::Error(TEXT("Start Play first"));
        FVector4f B(FCString::Atof(*A[0]), FCString::Atof(*A[1]), FCString::Atof(*A[2]), FCString::Atof(*A[3]));
        bool Visible = FCString::Atoi(*A[4]) != 0;
        for (int I = 0; I < 4; ++I) if (!FMath::IsFinite(B[I]) || B[I] < 0 || B[I] > 1) return FExecStatus::Error(TEXT("Box coordinates must be finite and normalized"));
        if (Visible && (B.Z <= B.X || B.W <= B.Y)) return FExecStatus::Error(TEXT("Box must have positive size"));
        return FExecStatus::OK(Weak->SetVisualDetection(B, Visible));
    }), TEXT("Show Python RGB detector result in HUD only"));
}

void APTZTrackingHUD::DrawHUD()
{
    Super::DrawHUD();
    if (!Canvas) return;
    if (!DisplayArena.IsValid()) for (TActorIterator<APTZArena> It(GetWorld()); It; ++It) { DisplayArena = *It; break; }
    if (!DisplayArena.IsValid()) return;
    const float W = FMath::Min(float(Canvas->SizeX), float(Canvas->SizeY) * 960 / 540);
    const float H = W * 540 / 960, X = (Canvas->SizeX - W) * 0.5f, Y = (Canvas->SizeY - H) * 0.5f;
    const float CX = X + W * 0.5f, CY = Y + H * 0.5f;
    const FLinearColor Green(0.1f, 1.0f, 0.3f), Blue(0.25f, 0.65f, 1.0f);
    DrawLine(CX - 10, CY, CX + 10, CY, Blue, 2); DrawLine(CX, CY - 10, CX, CY + 10, Blue, 2);
    DrawRect(FLinearColor(0.01f, 0.015f, 0.02f, 0.85f), X, Y, W, 48);
    DrawText(DisplayArena->GetDisplayMode(), FLinearColor::White, X + 12, Y + 5, nullptr, 0.85f);
    const bool Visible = DisplayArena->HasDetectionMessage() && DisplayArena->IsDetectionVisible();
    const FString Status = !DisplayArena->HasDetectionMessage() ? TEXT("Scene moves in preview; Python reset/step switches to synchronous time") :
        (Visible ? TEXT("RGB detector target visible | green: selected detection | blue: view center") : TEXT("RGB detector: no selected target"));
    DrawText(Status, Visible ? Green : FLinearColor::Yellow, X + 12, Y + 26, nullptr, 0.85f);
    if (!Visible) return;
    const FVector4f B = DisplayArena->GetDetectionBox();
    const float L = X + B.X * W, T = Y + B.Y * H, R = X + B.Z * W, Bottom = Y + B.W * H;
    DrawLine(CX, CY, (L + R) * 0.5f, (T + Bottom) * 0.5f, FLinearColor::Yellow, 2);
    DrawLine(L, T, R, T, Green, 2); DrawLine(R, T, R, Bottom, Green, 2);
    DrawLine(R, Bottom, L, Bottom, Green, 2); DrawLine(L, Bottom, L, T, Green, 2);
    DrawRect(Green, (L + R) * 0.5f - 3, (T + Bottom) * 0.5f - 3, 6, 6);
}

APTZGameMode::APTZGameMode()
{
    HUDClass = APTZTrackingHUD::StaticClass();
    DefaultPawnClass = nullptr; // Fixed PTZ camera; do not render a spare sphere pawn.
}
void APTZGameMode::StartPlay()
{
    Super::StartPlay();
    // The saved project map contains the editor-visible arena. Avoid a duplicate.
    for (TActorIterator<APTZArena> It(GetWorld()); It; ++It) return;
    GetWorld()->SpawnActor<APTZArena>();
}
