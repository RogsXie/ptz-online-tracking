"""Restore private release assets using existing GitHub credentials and hashes."""
import hashlib,json,os,ssl,subprocess,urllib.parse,urllib.request
from pathlib import Path
import certifi

ROOT=Path(__file__).resolve().parents[1]

class Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        result=super().redirect_request(req,fp,code,msg,headers,newurl)
        if result and urllib.parse.urlparse(req.full_url).netloc!=urllib.parse.urlparse(newurl).netloc:
            result.remove_header('Authorization')
        return result

def token():
    value=os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if value:return value
    environment=dict(os.environ,GCM_INTERACTIVE='Never',GIT_TERMINAL_PROMPT='0')
    result=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,env=environment,timeout=30)
    fields=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    if not fields.get('password'):raise RuntimeError('GitHub authentication required: configure Git credentials or GH_TOKEN.')
    return fields['password']

def main():
    manifest=json.loads((ROOT/'release-assets.json').read_text())
    auth=token();opener=urllib.request.build_opener(Redirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where())))
    headers={'Authorization':'Bearer '+auth,'User-Agent':'ptz-online-tracking-restore','Accept':'application/vnd.github+json'}
    url=f'https://api.github.com/repos/{manifest["repository"]}/releases/tags/{manifest["tag"]}'
    with opener.open(urllib.request.Request(url,headers=headers),timeout=60) as response:release=json.load(response)
    assets={asset['name']:asset for asset in release['assets']}
    for item in manifest['assets']:
        target=(ROOT/item['path']).resolve()
        if not target.is_relative_to(ROOT):raise ValueError('Asset path outside checkout')
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']:
            print('Verified existing:',item['path']);continue
        target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_suffix(target.suffix+'.download')
        download_headers=dict(headers,Accept='application/octet-stream')
        digest=hashlib.sha256();size=0
        with opener.open(urllib.request.Request(assets[item['name']]['url'],headers=download_headers),timeout=120) as response,temporary.open('wb') as output:
            while chunk:=response.read(1048576):output.write(chunk);digest.update(chunk);size+=len(chunk)
        if size!=item['bytes'] or digest.hexdigest()!=item['sha256']:raise RuntimeError('Asset checksum mismatch: '+item['name'])
        temporary.replace(target);print('Restored and verified:',item['path'])

if __name__=='__main__':main()
