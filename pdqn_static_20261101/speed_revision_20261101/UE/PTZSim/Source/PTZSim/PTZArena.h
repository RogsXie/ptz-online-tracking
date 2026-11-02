#pragma once

#include "CoreMinimal.h"
#include "HAL/PlatformMemory.h"
#include "GameFramework/Actor.h"
#include "GameFramework/GameModeBase.h"
#include "GameFramework/HUD.h"
#include "PTZArena.generated.h"

class AFusionCameraActor;
class ACameraActor;
class UStaticMeshComponent;
class UMaterialInterface;
class UFusionCamSensor;
class FJsonObject;
class ULitCamSensor;
struct FPTZPendingGPUFrame;

struct FPTZTargetRuntime
{
    TWeakObjectPtr<AActor> Actor;
    FString Id;
    TSharedPtr<FJsonObject> Path;
    FVector Origin = FVector::ZeroVector;
    double SpeedScale = 1;
    double Phase = 0;
    TArray<TWeakObjectPtr<UStaticMeshComponent>> Wheels;
    FVector LastVisualPosition = FVector::ZeroVector;
    double WheelAngle = 0;
    bool bVisualPositionValid = false;
};

/** UE replacement platform, with an editor-visible preview and seeded JSON scenarios. */
UCLASS()
class PTZSIM_API APTZArena : public AActor
{
    GENERATED_BODY()
public:
    APTZArena();
    virtual void OnConstruction(const FTransform& Transform) override;
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type Reason) override;
    virtual void Tick(float DeltaSeconds) override;
    FString ResetEpisode(int32 Seed);
    FString StepControl(float Pan, float Tilt, float Zoom, float Duration);
    void SetRealtime(bool bEnabled);
    FString PlayControl(float Pan, float Tilt, float Zoom, float Duration, bool CompactReply = false);
    TArray<uint8> CaptureRealtimeFrame();
    FString CaptureSharedFrame();
    FString SetAsyncCaptureRate(float Rate);
    FString CaptureAsyncSharedFrame();
    FString GetRealtimeTiming() const;
    FString GetState() const;
    bool LoadScenario(const FString& Filename, FString& Error);
    void SetDemo(bool bEnabled);
    void SetPreview(bool bEnabled);
    void SetMode(const FString& Mode);
    FString GetDisplayMode() const;
    FString SetVisualDetection(FVector4f Box, bool bVisible);
    bool HasDetectionMessage() const { return bDetectionReceived; }
    bool IsDetectionVisible() const { return bVisualDetectionVisible; }
    FVector4f GetDetectionBox() const { return VisualDetectionBox; }

    /** Relative to project directory, or an absolute JSON path. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="PTZ Scenario")
    FString ScenarioFile = TEXT("../paper_reproduction/scenarios/multi_crossing.json");

private:
    UPROPERTY() TObjectPtr<AFusionCameraActor> CaptureCamera;
    UPROPERTY() TObjectPtr<ACameraActor> ViewCamera;
    UPROPERTY() TObjectPtr<UMaterialInterface> ShapeMaterial;
    UPROPERTY() TObjectPtr<UStaticMesh> CubeMesh;
    UPROPERTY() TObjectPtr<UStaticMesh> SphereMesh;
    UPROPERTY() TArray<TObjectPtr<AActor>> SceneActors;
    UPROPERTY(VisibleAnywhere, Category="PTZ Editor Preview")
    TArray<TObjectPtr<UStaticMeshComponent>> PreviewParts;

    UStaticMeshComponent* AddPart(AActor* PartOwner, const TCHAR* Name, UStaticMesh* Mesh,
        FVector Location, FVector Scale, FLinearColor Color);
    AActor* SpawnSceneActor(const FString& Name);
    UFusionCamSensor* Sensor() const;
    void BuildScenario(FRandomStream& Random);
    void ApplyEnvironment();
    FVector TargetPosition(const FPTZTargetRuntime& Target, double Time) const;
    void MoveVehicles(float VisualDeltaSeconds = 0);
    void UpdateCamera();
    void RegisterCommands();
    TSharedRef<FJsonObject> TargetState(const FPTZTargetRuntime& Target) const;

    TSharedPtr<FJsonObject> ScenarioConfig;
    TSharedPtr<FJsonObject> ResolvedConfig;
    TArray<FPTZTargetRuntime> Targets;
    FPlatformMemory::FSharedMemoryRegion* SharedFrameRegion = nullptr;
    FString SharedFrameName;
    uint64 SharedFrameSequence = 0;
    void PumpAsyncFrames();
    TWeakObjectPtr<ULitCamSensor> AsyncLitSensor;
    TArray<TSharedPtr<FPTZPendingGPUFrame, ESPMode::ThreadSafe>> AsyncPending;
    TArray<TSharedPtr<FPTZPendingGPUFrame, ESPMode::ThreadSafe>> AsyncFree;
    TArray<uint8> AsyncLatestPacket;
    uint64 AsyncIssuedSequence = 0, AsyncReadySequence = 0;
    double AsyncRate = 0, AsyncNextCapture = 0, AsyncLatestCaptureTime = 0;
    FPlatformMemory::FSharedMemoryRegion* AsyncStreamRegion = nullptr;
    FString AsyncStreamName;
    bool AsyncDisabledBackgroundViewport = false;
    FVector CameraLocation = FVector(-2200, 0, 900);
    FRotator CameraRotation = FRotator(-12, 0, 0);
    FVector2D YawLimits = FVector2D(-40, 40);
    FVector2D PitchLimits = FVector2D(-25, 25);
    float PanSpeed = 25, TiltSpeed = 25, ZoomFOVSpeed = 5;
    float HorizontalFOV = 60, MinFOV = 4.4085f, MaxFOV = 60;
    double SimTime = 0;
    int32 EpisodeSeed = 0;
    bool bDemo = false;
    bool bFreePreview = true;
    bool bRealtimeControl = false;
    float RealtimePan = 0, RealtimeTilt = 0, RealtimeZoom = 0;
    double RealtimeActionRemaining = 0;
    double RealtimeElapsed = 0;
    TArray<double> RealtimeFrameIntervals;
    bool bLegacyTargetCenterReset = false;
    bool bDetectionReceived = false;
    bool bVisualDetectionVisible = false;
    FString ControlMode = TEXT("SCENE MOTION PREVIEW - FIXED CAMERA");
    FVector4f VisualDetectionBox = FVector4f(0, 0, 0, 0);
};

/** Draws Python-reported image detections; sensor RGB remains unannotated. */
UCLASS()
class PTZSIM_API APTZTrackingHUD : public AHUD
{
    GENERATED_BODY()
public:
    virtual void DrawHUD() override;
private:
    TWeakObjectPtr<APTZArena> DisplayArena;
};

UCLASS()
class PTZSIM_API APTZGameMode : public AGameModeBase
{
    GENERATED_BODY()
public:
    APTZGameMode();
    virtual void StartPlay() override;
};
