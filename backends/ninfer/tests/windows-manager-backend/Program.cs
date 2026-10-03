using NInfer.Manager;
using System.Buffers.Binary;
using System.Text.Json;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Logging;
using System.Security.AccessControl;
using System.Security.Principal;

var scratch = Path.Combine(AppContext.BaseDirectory, "fixtures-" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(scratch);
var defaults = Path.Combine(scratch, "defaults");
Directory.CreateDirectory(Path.Combine(defaults, "profiles"));
Directory.CreateDirectory(Path.Combine(scratch, "model"));
var modelPath = Path.Combine(scratch, "model", "test.ninfer");
Directory.CreateDirectory(Path.Combine(scratch, "engine"));
File.WriteAllBytes(Path.Combine(scratch, "engine", "ninfer-serve.exe"), []);
var id = Enumerable.Range(0, 16).Select(i => (byte)i).ToArray();
var checks = 0;
void Check(bool condition, string label) { if (!condition) throw new Exception("FAIL: " + label); checks++; Console.WriteLine("PASS " + label); }
void Reject(Action action, string label) { try { action(); } catch (Exception ex) when (ex is ArgumentException or InvalidDataException or JsonException or KeyNotFoundException) { Check(true, label); return; } throw new Exception("FAIL accepted: " + label); }
void Fixture(bool multipart)
{
    var directory = new { components = new { text = new { config = new { architectures = new[] { "Qwen3_5ForCausalLM" }, max_position_embeddings = 262144 }, resources = new Dictionary<string,string> { ["chat_template.jinja"] = "resource/text/chat_template.jinja" } } }, metadata = new { name = "Metadata fixture" }, objects = new[] { new { id = "resource/text/chat_template.jinja", kind = "resource", offset = 0, bytes = 16 } }, files = multipart ? new[] { new { path = (string?)null, payload_bytes = 32 }, new { path = (string?)"test.part", payload_bytes = 32 } } : new[] { new { path = (string?)null, payload_bytes = 32 } } };
    var json = JsonSerializer.SerializeToUtf8Bytes(directory);
    using(var stream = File.Create(modelPath)) { stream.Write("NINFER\0\x03"u8); stream.Write(BitConverter.GetBytes((ulong)json.Length)); stream.Write(id); stream.Write(json); stream.SetLength(((32L + json.Length + 4095) / 4096) * 4096 + 32); }
    if(multipart) using(var stream = File.Create(Path.Combine(scratch,"model","test.part"))) { stream.Write("NINPRT\0\x03"u8); stream.Write(BitConverter.GetBytes(1UL)); stream.Write(id); stream.SetLength(4096 + 32); }
}
Fixture(false);
var profile = new LaunchProfile { Id = "xxs-160k", Name = "Test", ModelPath = "model/test.ninfer", Parameters = new() { ["--max-context"]="163840",["--kv-capacity"]="163840",["--port"]="18081",["--default-max-tokens"]="0",["--model-id"]="A \"quoted\" model",["--preserve-thinking"]=null,["--chat-template"]="config/chat_template.jinja",["--device-profile-path"]="config/device-profiles.json" } };
File.WriteAllText(Path.Combine(defaults,"profiles","xxs.json"), JsonSerializer.Serialize(profile,ConfigurationStore.Json));
File.WriteAllText(Path.Combine(defaults,"settings.json"), JsonSerializer.Serialize(new ManagerSettings { StartWithWindows=true }, ConfigurationStore.Json));
File.WriteAllText(Path.Combine(defaults,"chat_template.jinja"), "fixture-template");
File.WriteAllText(Path.Combine(defaults,"chat_template.LICENSE"), "fixture-template-license");
File.WriteAllText(Path.Combine(defaults,"device-profiles.json"), "{}");
var store = new ConfigurationStore(ManagerPaths.Select(scratch, ""),defaults);
Check(store.Settings.StartWithWindows && store.Profiles.Count==1 && store.Profiles[0].Parameters["--max-context"]=="163840","default settings and XXS 160K import");
Check(File.ReadAllText(Path.Combine(scratch,"config","chat_template.jinja"))=="fixture-template","missing template copied");
Check(File.ReadAllText(Path.Combine(scratch,"config","chat_template.LICENSE"))=="fixture-template-license","template license is seeded beside the template");
File.WriteAllText(Path.Combine(scratch,"config","chat_template.jinja"),"user-template");
var reloaded = new ConfigurationStore(ManagerPaths.Select(scratch, ""),defaults);
Check(File.ReadAllText(Path.Combine(scratch,"config","chat_template.jinja"))=="user-template","existing template preserved");
var spec = store.BuildLaunchSpec("xxs-160k");
Check(spec.Executable==Path.Combine(scratch,"engine","ninfer-serve.exe") && spec.Arguments.Contains(Path.Combine(scratch,"config","chat_template.jinja")) && spec.Arguments.Contains(Path.Combine(scratch,"config","device-profiles.json")),"launch resolves engine and config resources from the package root");
Check(spec.Arguments.Contains("A \"quoted\" model") && spec.Arguments.Contains("--preserve-thinking") && spec.ApiBase=="http://127.0.0.1:18081/v1","token array preserves quoted model ID and flag");
Check(spec.RequestLogPath.StartsWith(Path.Combine(scratch,"logs")) && spec.Arguments.Contains(spec.RequestLogPath),"owned unique request log");
var clone = store.Profiles[0]; clone.Parameters["--max-context"]="1";
Check(store.Profiles[0].Parameters["--max-context"]=="163840","returned profiles cannot mutate stored configuration");
store.SaveProfile(profile with { Name="Updated" });
Check(Directory.GetFiles(Path.Combine(scratch,"config","history")).Length==1 && new ConfigurationStore(ManagerPaths.Select(scratch, ""),defaults).Profiles[0].Name=="Updated","atomic save + history + reload");
foreach(var policy in new[] { "default", "mixed", "strict", "strict-64-128", "strict-0-1", "strict-96-256", "strict-17592186044415-16384" })
{
 var candidate=profile with {Parameters=new(profile.Parameters)};
 candidate.Parameters["--cuda-memory-policy"]=policy;
 candidate.Parameters["--host-cache-mib"]="6144";
 store.SaveProfile(candidate);
 var expected=policy=="strict-64-128"?"strict":policy;
 var persisted=new ConfigurationStore(ManagerPaths.Select(scratch, ""),defaults).Profiles.Single(p=>p.Id==profile.Id);
 var launch=store.BuildLaunchSpec(profile.Id);
 var policyIndex=launch.Arguments.ToList().IndexOf("--cuda-memory-policy");
 Check(persisted.Parameters["--cuda-memory-policy"]==expected&&persisted.Parameters["--host-cache-mib"]=="6144"&&policyIndex>=0&&launch.Arguments[policyIndex+1]==expected,"memory policy saves, reloads and launches as one option: "+policy);
}
var hybridProfile=profile with {Parameters=new(profile.Parameters)};
hybridProfile.Parameters["--cuda-memory-policy"]="strict";
hybridProfile.Parameters["--device-snapshot-slots"]="1";
hybridProfile.Parameters["--host-cache-mib"]="0";
store.SaveProfile(hybridProfile);
Check(!store.BuildLaunchSpec(profile.Id).Arguments.Contains("--use-alt-prefix-caching")&&new ConfigurationStore(ManagerPaths.Select(scratch, ""),defaults).Profiles.Single(p=>p.Id==profile.Id).Parameters["--host-cache-mib"]=="0","strict enables hybrid semantics without an injected flag and Host cache can independently be zero");
foreach(var policy in new[]{"default","mixed"})
{
 var candidate=hybridProfile with {Parameters=new(hybridProfile.Parameters)};
 candidate.Parameters["--cuda-memory-policy"]=policy;
 Reject(()=>store.SaveProfile(candidate),policy+" rejects Hybrid-only parameters without the explicit cache flag");
 candidate.Parameters["--use-alt-prefix-caching"]=null;
 candidate.Parameters["--kv-capacity"]="auto";
 store.SaveProfile(candidate);
 var persisted=new ConfigurationStore(ManagerPaths.Select(scratch, ""),defaults).Profiles.Single(p=>p.Id==profile.Id);
 Check(persisted.Parameters["--cuda-memory-policy"]==policy&&persisted.Parameters.ContainsKey("--use-alt-prefix-caching")&&persisted.Parameters["--device-snapshot-slots"]=="1"&&store.BuildLaunchSpec(profile.Id).Arguments.Contains("auto"),policy+" preserves explicit hybrid cache and auto capacity without inserting strict settings");
}
foreach(var policy in new[]{"mixed","strict"})
{
 foreach(var option in new[]{"--kv-headroom-mib","--vram-headroom-mib","--vision","--vision-cpu","--wddm-evictable-budget","--devices"})
 {
  var candidate=profile with {Parameters=new(profile.Parameters)};
  candidate.Parameters["--cuda-memory-policy"]=policy;
  candidate.Parameters["--kv-capacity"]="auto";
  candidate.Parameters[option]=option=="--devices"?"0,1":option.EndsWith("headroom-mib",StringComparison.Ordinal)?"64":null;
  Reject(()=>store.SaveProfile(candidate),policy+" rejects incompatible option "+option);
 }
 var singleDevice=profile with {Parameters=new(profile.Parameters)};
 singleDevice.Parameters["--cuda-memory-policy"]=policy;
 singleDevice.Parameters["--devices"]="0";
 store.SaveProfile(singleDevice);
 var launch=store.BuildLaunchSpec(profile.Id);
 var deviceIndex=launch.Arguments.ToList().IndexOf("--devices");
 Check(deviceIndex>=0&&launch.Arguments[deviceIndex+1]=="0",policy+" accepts one explicitly selected GPU");
}
foreach(var headroom in new[]{"--kv-headroom-mib","--vram-headroom-mib"})
{
 var candidate=profile with {Parameters=new(profile.Parameters)};
 candidate.Parameters["--cuda-memory-policy"]="default";
 candidate.Parameters["--kv-capacity"]="auto";
 candidate.Parameters[headroom]="96";
 store.SaveProfile(candidate);
 var launch=store.BuildLaunchSpec(profile.Id);
 var index=launch.Arguments.ToList().IndexOf(headroom);
 Check(index>=0&&launch.Arguments[index+1]=="96","default retains explicit auto-capacity headroom: "+headroom);
}
var storedBeforeInvalid=File.ReadAllText(Path.Combine(scratch,"config","profiles",profile.Id+".json"));
var revisionsBeforeInvalid=Directory.GetFiles(Path.Combine(scratch,"config","history")).Length;
foreach(var policy in new string?[]{null,"","bogus","STRICT","strict-64","strict-64-128-extra","strict-64-128\n","strict--1-128","strict-1.5-128","strict-64-0","strict-64-16385","strict-17592186044416-128","strict-18446744073709551616-1"})
{
 var candidate=profile with {Parameters=new(profile.Parameters)};candidate.Parameters["--cuda-memory-policy"]=policy;
 Reject(()=>store.SaveProfile(candidate),"invalid memory policy rejected: "+(policy??"null"));
}
foreach(var option in new[]{"--use-original-prefix-caching","--no-prefix-reuse"})
{
 var candidate=hybridProfile with {Parameters=new(hybridProfile.Parameters)};candidate.Parameters[option]=null;
 Reject(()=>store.SaveProfile(candidate),"strict rejects incompatible cache choice: "+option);
}
Check(File.ReadAllText(Path.Combine(scratch,"config","profiles",profile.Id+".json"))==storedBeforeInvalid&&Directory.GetFiles(Path.Combine(scratch,"config","history")).Length==revisionsBeforeInvalid,"rejected memory policies do not change saved parameters or create revisions");
LaunchProfile WithOptions(params (string Key, string? Value)[] options)
{
 var candidate=profile with {Parameters=new(profile.Parameters)};
 foreach(var (key,value) in options)candidate.Parameters[key]=value;
 return candidate;
}
string? ArgumentValue(LaunchSpec launch,string option)
{
 var index=launch.Arguments.ToList().IndexOf(option);
 return index<0?null:launch.Arguments[index+1];
}
foreach(var backend in new[]{"mtp","dflash","dflash2"})
{
 var candidate=WithOptions(("--spec",backend),("--draft-tokens","15"),("--lm-head-draft",null),
  ("--ngram-draft-tokens","63"),("--ngram-min-match","64"),("--ngram-archive-mib","512"),
  ("--ngram-session-mib","128"),("--ngram-native-sessions",null),("--lookup-ngram","12"));
 if(backend=="mtp") {candidate.Parameters["--adaptive-mtp"]=null;candidate.Parameters["--mtp-attention-window"]="16";}
 store.SaveProfile(candidate);
 var persisted=new ConfigurationStore(ManagerPaths.Select(scratch,""),defaults).Profiles.Single(p=>p.Id==profile.Id);
 var launch=store.BuildLaunchSpec(profile.Id);
 Check(candidate.Parameters.All(pair=>persisted.Parameters.ContainsKey(pair.Key)&&persisted.Parameters[pair.Key]==pair.Value)&&
  ArgumentValue(launch,"--spec")==backend&&ArgumentValue(launch,"--draft-tokens")=="15"&&
  ArgumentValue(launch,"--ngram-session-mib")=="128"&&launch.Arguments.Contains("--ngram-native-sessions"),
  "speculative configuration saves, reloads and launches with values and switches intact: "+backend);
}
foreach(var width in new[]{"1","2","3","15"})
{
 var candidate=WithOptions(("--spec","mtp"),("--draft-tokens",width),("--adaptive-mtp",null),("--max-concurrency","8"));
 store.SaveProfile(candidate);
 Check(!store.BuildLaunchSpec(profile.Id).Arguments.Contains("--ngram-draft-tokens"),"adaptive MTP supports draft width "+width+" and leaves the server's concurrency-safe default ngram width implicit");
}
var disabledDrafting=WithOptions(("--draft-tokens","0"),("--ngram-draft-tokens","0"),("--mtp-attention-window","0"),("--lookup-ngram","8"),("--ngram-session-mib","0"));
store.SaveProfile(disabledDrafting);
Check(ArgumentValue(store.BuildLaunchSpec(profile.Id),"--lookup-ngram")=="8","disabled speculative decoding preserves a dormant lookup value and zero disabled capacities");
foreach(var pair in new (string,string?)[]{("--spec",null),("--spec","off"),("--draft-tokens","1"),("--lm-head-draft",null),
 ("--adaptive-mtp",null),("--mtp-attention-window","8192"),("--ngram-draft-tokens","1"),("--ngram-native-sessions",null)})
 Reject(()=>store.SaveProfile(WithOptions(pair)),"inactive speculative backend rejects dependent option "+pair.Item1);
foreach(var pair in new (string,string?)[]{("--draft-tokens","0"),("--draft-tokens","16"),("--draft-tokens",null),
 ("--ngram-draft-tokens","64"),("--ngram-draft-tokens","-1"),("--ngram-min-match","3"),("--ngram-min-match","65"),
 ("--mtp-attention-window","2"),("--lookup-ngram","-1"),("--ngram-archive-mib","17592186044416"),
 ("--ngram-session-mib",null),("--lm-head-draft","true"),("--adaptive-mtp","false"),("--ngram-native-sessions","true")})
{
 var candidate=WithOptions(("--spec","mtp"),("--draft-tokens","2"));candidate.Parameters[pair.Item1]=pair.Item2;
 Reject(()=>store.SaveProfile(candidate),"invalid speculative value rejected: "+pair.Item1+"="+(pair.Item2??"null"));
}
Reject(()=>store.SaveProfile(WithOptions(("--spec","mtp"))),"enabled backend requires explicit positive draft width");
foreach(var backend in new[]{"dflash","dflash2"})
 foreach(var pair in new (string,string?)[]{("--adaptive-mtp",null),("--mtp-attention-window","8192")})
  Reject(()=>store.SaveProfile(WithOptions(("--spec",backend),("--draft-tokens","2"),pair)),backend+" rejects MTP-only option "+pair.Item1);
Reject(()=>store.SaveProfile(WithOptions(("--spec","mtp"),("--draft-tokens","2"),("--ngram-draft-tokens","16"),("--max-concurrency","2"))),"wide ngram drafting requires single concurrency");
foreach(var options in new (string,string?)[][]{
 [("--ngram-archive-mib","64")],
 [("--ngram-archive-mib","128"),("--ngram-session-mib","0")],
 [("--ngram-archive-mib","128"),("--ngram-session-mib","129")],
 [("--ngram-archive-mib","128"),("--ngram-draft-tokens","0")]})
{
 var candidate=WithOptions(("--spec","mtp"),("--draft-tokens","2"));foreach(var pair in options)candidate.Parameters[pair.Item1]=pair.Item2;
 Reject(()=>store.SaveProfile(candidate),"ngram archive enforces active copy drafting and a bounded positive session share");
}
foreach(var residency in new[]{"resident","overlay","cpu"})
{
 var candidate=WithOptions(("--vision",null),("--vision-residency",residency),("--media-cache-mib","0"),
  ("--media-live-mib","4096"),("--media-preprocess-threads","64"));
 if(residency=="cpu")candidate.Parameters["--vision-cpu"]=null;
 store.SaveProfile(candidate);
 var persisted=new ConfigurationStore(ManagerPaths.Select(scratch,""),defaults).Profiles.Single(p=>p.Id==profile.Id);
 var launch=store.BuildLaunchSpec(profile.Id);
 Check(ArgumentValue(launch,"--vision-residency")==residency&&ArgumentValue(launch,"--media-preprocess-threads")=="64"&&
  persisted.Parameters["--media-cache-mib"]=="0"&&!launch.Arguments.Contains("--vision-max-merged"),
  "Vision "+residency+" preserves media settings and the server's implicit merged-token limit");
}
var cpuVision=WithOptions(("--vision-cpu",null),("--vision-max-merged","16384"));
store.SaveProfile(cpuVision);
Check(store.BuildLaunchSpec(profile.Id).Arguments.Contains("--vision-cpu")&&ArgumentValue(store.BuildLaunchSpec(profile.Id),"--vision-max-merged")=="16384","CPU Vision shorthand enables Vision without an injected --vision and accepts an explicit limit above its 256 default");
foreach(var pair in new (string,string?)[]{("--vision","true"),("--vision-cpu","false"),("--vision-residency",null),
 ("--vision-residency","gpu"),("--vision-residency","overlay"),("--vision-residency","cpu"),
 ("--vision-max-merged","63"),("--vision-max-merged","16385"),("--media-cache-mib","-1"),
 ("--media-cache-mib","17592186044416"),("--media-live-mib","0"),("--media-live-mib",null),("--media-preprocess-threads","65")})
 Reject(()=>store.SaveProfile(WithOptions(pair)),"invalid Vision/media option rejected: "+pair.Item1+"="+(pair.Item2??"null"));
foreach(var residency in new[]{"resident","overlay"})
 Reject(()=>store.SaveProfile(WithOptions(("--vision-cpu",null),("--vision-residency",residency))),"CPU Vision rejects order-dependent conflicting residency "+residency);
Reject(()=>store.SaveProfile(WithOptions(("--vision",null),("--vision-residency","cpu"),("--vision-offload","on"))),"Vision alias cannot silently override the selected residency");
foreach(var pathOption in new[]{"--request-log-jsonl","--chat-template","--device-profile-path","--context-cost-presets","--disk-kv-path","--prefix-cache-file"})
 foreach(var invalidPath in new string?[]{null,""})Reject(()=>store.SaveProfile(WithOptions((pathOption,invalidPath))),"path parameter requires a nonempty value: "+pathOption);
foreach(var options in new (string,string?)[][]{
 [("--stats-port","18081")],[("--stats-port","65536")],[("--stats-port",null)],[("--api-key",null)],[("--api-key","line\nbreak")]})
 Reject(()=>store.SaveProfile(WithOptions(options)),"invalid telemetry listener or authentication option rejected");
store.SaveProfile(WithOptions(("--stats-port",store.Settings.WebPort.ToString())));
Reject(()=>store.BuildLaunchSpec(profile.Id),"dedicated statistics listener cannot occupy the management website port");
store.SaveProfile(WithOptions(("--stats-port","0"),("--api-key","")));
Check(ArgumentValue(store.BuildLaunchSpec(profile.Id),"--api-key")==""&&ArgumentValue(store.BuildLaunchSpec(profile.Id),"--stats-port")=="0","empty API key and zero stats port preserve explicitly disabled behavior");
store.SaveProfile(profile);
Reject(()=>store.SaveProfile(profile with {Id="../escape"}),"profile path traversal rejected");
foreach(var pair in new[]{("--max-context","0"),("--kv-capacity","8192"),("--default-max-tokens","163841"),("--port","65536"),("--max-context", "2147483648")}) {
 var p = profile with { Parameters=new(profile.Parameters) }; p.Parameters[pair.Item1]=pair.Item2; Reject(()=>store.SaveProfile(p),pair.Item1+" invalid value rejected");
}
store.SaveProfile(profile with {Id="secondary"});
store.DeleteProfile("secondary");
Check(new ConfigurationStore(ManagerPaths.Select(scratch, ""),defaults).Profiles.Count==1,"deleted profile remains deleted across restart");
var embeddedRoot=Path.Combine(scratch,"embedded-package");
var embedded=new ConfigurationStore(ManagerPaths.Select(embeddedRoot, ""));
Check(embedded.Profiles.Count==2&&!embedded.Settings.StartWithWindows&&embedded.Profiles.Single(p=>p.Id=="xxs-160k").Parameters["--max-context"]=="163840"&&embedded.Profiles.Single(p=>p.Id=="s-128k").Parameters["--max-context"]=="131072","embedded resources initialize the complete package configuration without an external seed directory or implicit startup registration");
Check(embedded.Profiles.All(p=>p.EnginePath=="engine/ninfer-serve.exe"&&p.Parameters["--chat-template"]=="config/chat_template.jinja"&&p.Parameters["--device-profile-path"]=="config/device-profiles.json")&&File.Exists(Path.Combine(embeddedRoot,"config","chat_template.LICENSE")),"embedded launch profiles and template license use the package layout");
embedded.SaveSettings(embedded.Settings with {Language="en",AutoStartModel=false});
embedded.SaveProfile(embedded.Profiles.Single(p=>p.Id=="xxs-160k") with {Name="Saved profile"});
embedded.DeleteProfile("s-128k");
File.WriteAllText(Path.Combine(embeddedRoot,"config","chat_template.jinja"),"saved-template");
File.WriteAllText(Path.Combine(embeddedRoot,"config","device-profiles.json"),"{\"saved\":true}");
var embeddedReloaded=new ConfigurationStore(ManagerPaths.Select(embeddedRoot, ""));
Check(embeddedReloaded.Settings is {Language:"en",AutoStartModel:false}&&embeddedReloaded.Profiles is {Count:1}&&embeddedReloaded.Profiles[0].Name=="Saved profile"&&File.ReadAllText(Path.Combine(embeddedRoot,"config","chat_template.jinja"))=="saved-template"&&File.ReadAllText(Path.Combine(embeddedRoot,"config","device-profiles.json"))=="{\"saved\":true}","restarting preserves saved preferences, profile edits, deleted seed profiles and customized resources");
File.Delete(Path.Combine(embeddedRoot,"config","chat_template.LICENSE"));
_=new ConfigurationStore(ManagerPaths.Select(embeddedRoot, ""));
Check(File.ReadAllText(Path.Combine(embeddedRoot,"config","chat_template.LICENSE")).Contains("Apache License"),"missing template license can be restored from the executable");
var packageSeedRoot=Path.Combine(scratch,"installed-package");
var packageSeedStore=new ConfigurationStore(ManagerPaths.Select(packageSeedRoot,""),defaults);
packageSeedStore.SaveProfile(profile with {Name="Packaged settings"});
packageSeedStore.SaveSettings(packageSeedStore.Settings with {Language="en",AutoStartModel=false});
Directory.CreateDirectory(Path.Combine(packageSeedRoot,"model"));
Directory.CreateDirectory(Path.Combine(packageSeedRoot,"engine"));
File.Copy(modelPath,Path.Combine(packageSeedRoot,"model","test.ninfer"));
File.WriteAllBytes(Path.Combine(packageSeedRoot,"engine","ninfer-serve.exe"),[]);
var packagedTemplate=Path.Combine(packageSeedRoot,"config","chat_template.jinja");
File.WriteAllText(packagedTemplate,"packaged-template");
File.SetAttributes(packagedTemplate,FileAttributes.ReadOnly);
var packageFilesBefore=Directory.GetFiles(packageSeedRoot,"*",SearchOption.AllDirectories).ToDictionary(path=>path,File.ReadAllBytes);
var localRoot=Path.Combine(scratch,"local-app-data");
var paths=ManagerPaths.Select(packageSeedRoot,localRoot);
Check(!paths.IsPortable&&paths.DataRoot.StartsWith(Path.Combine(localRoot,"NInferManager"))&&paths.InstallationId==ManagerPaths.Select(packageSeedRoot.ToUpperInvariant()+Path.DirectorySeparatorChar,localRoot).InstallationId,"LocalAppData is preferred and package identity normalizes case and trailing separators");
var perUser=new ConfigurationStore(paths);
Check(perUser.Settings is {Language:"en",AutoStartModel:false}&&perUser.Profiles.Single().Name=="Packaged settings"&&File.ReadAllText(Path.Combine(paths.ConfigRoot,"chat_template.jinja"))=="packaged-template"&&Directory.GetFiles(Path.Combine(paths.ConfigRoot,"history")).Length==2,"first launch seeds existing package preferences, profile, history and resources into user data");
File.WriteAllText(Path.Combine(paths.ConfigRoot,"context-costs.json"),"{}");
var customPaths=WithOptions(("--cuda-memory-policy","strict"),("--context-cost-presets","config/context-costs.json"),
 ("--device-profile-path","config/new-calibration.json"),("--prefix-cache-file","cache/prefix.bin"),
 ("--request-log-jsonl","logs/custom/server.jsonl"),("--request-log-max-mib","64"),("--request-log-keep","3"));
perUser.SaveProfile(customPaths);
var customLaunch=perUser.BuildLaunchSpec(profile.Id);
Check(ArgumentValue(customLaunch,"--context-cost-presets")==Path.Combine(paths.ConfigRoot,"context-costs.json")&&
 ArgumentValue(customLaunch,"--device-profile-path")==Path.Combine(paths.ConfigRoot,"new-calibration.json")&&
 ArgumentValue(customLaunch,"--prefix-cache-file")==Path.Combine(paths.DataRoot,"cache","prefix.bin"),
 "context presets use user config, new device profiles need not exist and mutable prefix cache resolves under user data");
Check(customLaunch.RequestLogPath==Path.Combine(paths.LogsRoot,"custom","server.jsonl")&&
 ArgumentValue(customLaunch,"--request-log-jsonl")==customLaunch.RequestLogPath&&Directory.Exists(Path.GetDirectoryName(customLaunch.RequestLogPath))&&
 ArgumentValue(customLaunch,"--request-log-max-mib")=="64"&&ArgumentValue(customLaunch,"--request-log-keep")=="3",
 "custom relative request log and rotation values launch unchanged while telemetry follows the same writable absolute path");
var customLog=Path.Combine(scratch,"external-log","request.jsonl");
perUser.SaveProfile(WithOptions(("--disk-kv-path","cache/disk-kv"),("--request-log-jsonl",customLog)));
customLaunch=perUser.BuildLaunchSpec(profile.Id);
Check(customLaunch.RequestLogPath==customLog&&ArgumentValue(customLaunch,"--request-log-jsonl")==customLog&&
 ArgumentValue(customLaunch,"--disk-kv-path")==Path.Combine(paths.DataRoot,"cache","disk-kv"),
 "absolute custom log is preserved and relative disk KV directory uses the writable user data root");
perUser.SaveProfile(profile with {Name="Packaged settings"});
var userProfile=perUser.Profiles.Single() with {Name="User edits",Parameters=new(perUser.Profiles.Single().Parameters)};
userProfile.Parameters["--cuda-memory-policy"]="strict";
userProfile.Parameters["--prefix-cache-file"]="context-cache.bin";
perUser.SaveProfile(userProfile);
perUser.SaveSettings(perUser.Settings with {Language="zh"});
File.WriteAllText(Path.Combine(paths.ConfigRoot,"chat_template.jinja"),"user-data-template");
File.WriteAllText(Path.Combine(paths.ConfigRoot,"device-profiles.json"),"{\"user\":true}");
var perUserSpec=perUser.BuildLaunchSpec(profile.Id);
Check(perUserSpec.Executable==Path.Combine(packageSeedRoot,"engine","ninfer-serve.exe")&&perUserSpec.Arguments[0]==Path.Combine(packageSeedRoot,"model","test.ninfer")&&perUserSpec.WorkingDirectory==packageSeedRoot&&perUserSpec.Arguments.Contains(Path.Combine(paths.ConfigRoot,"chat_template.jinja"))&&perUserSpec.Arguments.Contains(Path.Combine(paths.ConfigRoot,"device-profiles.json"))&&perUserSpec.Arguments.Contains(Path.Combine(paths.DataRoot,"context-cache.bin"))&&perUserSpec.RequestLogPath.StartsWith(paths.LogsRoot),"launch reads engine/model from the package and writes resources, cache and logs under the selected data root");
Check(packageFilesBefore.Count==Directory.GetFiles(packageSeedRoot,"*",SearchOption.AllDirectories).Length&&packageFilesBefore.All(pair=>File.ReadAllBytes(pair.Key).SequenceEqual(pair.Value)),"saving user data never writes package files, including the read-only source template");
var userReload=new ConfigurationStore(ManagerPaths.Select(packageSeedRoot,localRoot));
Check(userReload.Settings.Language=="zh"&&userReload.Profiles.Single().Name=="User edits"&&File.ReadAllText(Path.Combine(paths.ConfigRoot,"chat_template.jinja"))=="user-data-template","existing LocalAppData settings and resources take priority over packaged seeds on restart");
var packageDirectory=new DirectoryInfo(packageSeedRoot);
var originalAccess=packageDirectory.GetAccessControl();
try
{
 var readOnlyAccess=packageDirectory.GetAccessControl();
 readOnlyAccess.AddAccessRule(new FileSystemAccessRule(WindowsIdentity.GetCurrent().User!,FileSystemRights.Write|FileSystemRights.Delete|FileSystemRights.DeleteSubdirectoriesAndFiles,InheritanceFlags.ContainerInherit|InheritanceFlags.ObjectInherit,PropagationFlags.None,AccessControlType.Deny));
 packageDirectory.SetAccessControl(readOnlyAccess);
 var protectedStore=new ConfigurationStore(ManagerPaths.Select(packageSeedRoot,localRoot));
 protectedStore.SaveSettings(protectedStore.Settings with {Language="en"});
 protectedStore.SaveProfile(userProfile with {Name="Saved with read-only package"});
 var protectedSpec=protectedStore.BuildLaunchSpec(profile.Id);
 Check(protectedSpec.Executable==Path.Combine(packageSeedRoot,"engine","ninfer-serve.exe")&&protectedSpec.RequestLogPath.StartsWith(paths.LogsRoot),"a package directory denied write/delete access can still load resources and save settings/profiles through LocalAppData");
}
finally
{
 var restoredAccess=new DirectorySecurity();
 restoredAccess.SetSecurityDescriptorBinaryForm(originalAccess.GetSecurityDescriptorBinaryForm(),AccessControlSections.Access);
 packageDirectory.SetAccessControl(restoredAccess);
}
var absoluteProfile=userProfile with {Parameters=new(userProfile.Parameters)};
absoluteProfile.Parameters["--chat-template"]=packagedTemplate;
perUser.SaveProfile(absoluteProfile);
Check(perUser.BuildLaunchSpec(profile.Id).Arguments.Contains(packagedTemplate),"explicit absolute resource paths are preserved");
var invalidUserSettings=Path.Combine(paths.ConfigRoot,"settings.json");
File.WriteAllText(invalidUserSettings,"BROKEN USER JSON");
Reject(()=>new ConfigurationStore(ManagerPaths.Select(packageSeedRoot,localRoot)),"invalid user configuration fails instead of falling back to packaged configuration");
Check(File.ReadAllText(invalidUserSettings)=="BROKEN USER JSON","invalid user configuration is preserved");
Check(ManagerPaths.Select(Path.Combine(scratch,"second-install"),localRoot).DataRoot!=paths.DataRoot,"different package copies use independent user data directories");
var blockedLocal=Path.Combine(scratch,"blocked-local");
File.WriteAllText(blockedLocal,"file blocks directory creation");
var portablePaths=ManagerPaths.Select(Path.Combine(scratch,"portable-package"),blockedLocal);
Check(portablePaths.IsPortable&&Directory.Exists(portablePaths.RuntimeRoot)&&Directory.Exists(portablePaths.LogsRoot),"unwritable LocalAppData falls back to a writable package data directory");
var blockedPackage=Path.Combine(scratch,"blocked-package");
File.WriteAllText(blockedPackage,"file blocks directory creation");
try { _=ManagerPaths.Select(blockedPackage,blockedLocal); throw new Exception("Both unwritable data roots accepted"); }
catch(IOException failure) { Check(failure.Message.Contains("Cannot create writable manager data")&&failure.Message.Contains(blockedPackage)&&failure.Message.Contains(blockedLocal),"both unavailable locations produce an actionable error listing the failed destinations"); }
File.SetAttributes(packagedTemplate,FileAttributes.Normal);
Reject(()=>store.DeleteProfile("xxs-160k"),"default profile deletion rejected");
Check(ModelCatalog.ReadModel(modelPath,scratch) is {IsComplete:true,MaxContext:262144,PartCount:1},"single volume metadata validation");
Fixture(true);
Check(ModelCatalog.ReadModel(modelPath,scratch) is {IsComplete:true,PartCount:2,FileBytes:8256},"continuation header and full declared file lengths");
File.Delete(Path.Combine(scratch,"model","test.part"));
Check(!ModelCatalog.ReadModel(modelPath,scratch).IsComplete,"missing continuation detected");
Fixture(true);
using(var stream=File.OpenWrite(Path.Combine(scratch,"model","test.part"))) { stream.Position=16; stream.WriteByte(255); }
Check(!ModelCatalog.ReadModel(modelPath,scratch).IsComplete,"foreign continuation artifact ID detected");
Fixture(false);
using(var stream=File.OpenWrite(modelPath)) { stream.Position=8; stream.Write(BitConverter.GetBytes(65UL*1024*1024)); }
Check(!ModelCatalog.ReadModel(modelPath,scratch).IsComplete,"oversized directory rejected before allocation");
Fixture(false);
var models = await new ModelCatalog(store).ScanAsync();
Check(models.Count==1&&models[0].IsComplete,"directory scan with cached model metadata");
if (Environment.GetEnvironmentVariable("NINFER_TEST_ARTIFACT") is { Length: > 0 } realPath)
{
    var real = ModelCatalog.ReadModel(realPath, Path.GetDirectoryName(realPath)!);
    Check(real.IsComplete && real.MaxContext > 0 && real.Components.Contains("text"), "real v3 artifact metadata without tensor reads or GPU");
}

File.WriteAllText(Path.Combine(scratch,"config","settings.json"),"BROKEN JSON");
Reject(()=>new ConfigurationStore(ManagerPaths.Select(scratch, ""),defaults),"bad settings JSON rejected");
Check(File.ReadAllText(Path.Combine(scratch,"config","settings.json"))=="BROKEN JSON","bad configuration preserved unchanged");
using(var record=JsonDocument.Parse("""{"event":"request_done","request":{"request_id":7},"result":{"prompt_tokens":100,"completion_tokens":11,"model_thinking_tokens":4,"prefix_cache_hit_tokens":10,"computed_prefill_tokens":90,"finish_reason":"stop"},"timings_seconds":{"ttft":0.25,"prefill":0.1,"decode":0.2}}""")) {
 var normalized=TelemetryReader.NormalizeRequest(record.RootElement);
 Check(normalized.GetProperty("request_id").GetInt32()==7 && normalized.GetProperty("reasoning_tokens").GetInt32()==4 && normalized.GetProperty("cached_tokens").GetInt32()==10,"flat UI token counters preserve raw request");
 Check(normalized.GetProperty("ttft_ms").GetDouble()==250 && normalized.GetProperty("prefill_tps").GetDouble()==900 && normalized.GetProperty("decode_tps").GetDouble()==50 && normalized.GetProperty("result").GetProperty("completion_tokens").GetInt32()==11,"flat TTFT and prefill/decode rates with original record intact");
}
using(var record=JsonDocument.Parse("""{"event":"request_error","request":{"request_id":8}}""")) Check(TelemetryReader.NormalizeRequest(record.RootElement).GetProperty("decode_tps").ValueKind==JsonValueKind.Null,"error records do not invent throughput");
using(var record=JsonDocument.Parse("""{"event":"request_done","timings_seconds":{"ttft":0.1},"engine_timing":{"queue_wait_seconds":0.2}}""")) Check(TelemetryReader.NormalizeRequest(record.RootElement).GetProperty("proc_ms").ValueKind==JsonValueKind.Null,"inconsistent queue and TTFT cannot invent zero processing time");
File.WriteAllText(Path.Combine(scratch,"config","settings.json"),JsonSerializer.Serialize(store.Settings,ConfigurationStore.Json));
var monitorNow = DateTimeOffset.UtcNow;
string RequestLine(int requestId, string kind = "request_done", long? timestamp = null) => JsonSerializer.Serialize(new Dictionary<string,object?> {
 ["event"]=kind, ["timestamp_unix_ms"]=timestamp??monitorNow.ToUnixTimeMilliseconds(), ["server_instance_id"]="fixture-server",
 ["request"]=new {request_id=requestId,protocol="openai"},
 ["result"]=kind=="request_done"?new {prompt_tokens=100,completion_tokens=11,model_thinking_tokens=4,prefix_cache_hit_tokens=25,computed_prefill_tokens=75,prefix_reuse_path="private_endpoint",finish_reason="stop_token"}:null,
 ["timings_seconds"]=new {ttft=.25,prefill=.5,decode=.2}, ["engine_timing"]=new {queue_wait_seconds=.05},
 ["speculative"]=new {accepted_tokens=6,drafted_tokens=10,rounds=5}, ["error"]=kind=="request_done"?null:new {message="Fixture failure"}
});
var monitorLog=Path.Combine(scratch,"monitor.jsonl");
File.WriteAllText(monitorLog,string.Join('\n',Enumerable.Range(1,55).Select(id=>RequestLine(id)))+"\n"+RequestLine(56,"request_error")+"\n"+RequestLine(57,"request_rejected")+"\n");
var run=new EngineSnapshot("Running",Pid:123,RequestLogPath:monitorLog,StartedAt:monitorNow);
var gpu=new GpuTelemetrySample(true,"Fixture GPU",16UL<<30,14UL<<30,2UL<<30,null,UtilizationPercent:75,PowerWatts:220,PowerLimitWatts:300,TemperatureC:62);
JsonElement EngineStats(double prefill=100,double decode=10,double transfer=0) => JsonSerializer.SerializeToElement(new {
 counters=new {computed_prefill_tokens=prefill,committed_decode_tokens=decode},
 memory=new {cuda_residency=new {dedicated_bytes=14L<<30}}, occupancy=new {device_main_kv_tokens=4096,host_kv_bytes=1024},
 capacity=new {kv_capacity_tokens=163840,host_kv_bytes=6442450944L},requests=new {waiting=0,running=1,prefilling=0,materializing=0},
 context_cache=new {main_kv_transfers=new {h2d=new {bytes=transfer,pages=1,seconds=.1},d2h=new {bytes=transfer,pages=1,seconds=.1},d2d=new {bytes=transfer,pages=1,seconds=.1}},
 backend_kv_transfers=new {h2d=new {bytes=transfer,pages=1,seconds=.1},d2h=new {bytes=transfer,pages=1,seconds=.1},d2d=new {bytes=transfer,pages=1,seconds=.1}}, pressure=new {spill_pages=2,private_owners_evicted=1}},
 scheduler=new {running=1,prefilling=0,waiting=0,decode_ready=1,materializing=0}
});
using(var monitor=new MonitorSession()) {
 monitor.UseEngine(run); monitor.ReadRequests();
 var first=monitor.Sample(EngineStats(),gpu,monitorNow);
 Check(first.Counts is {Completed:55,Errors:1,Rejected:1}&&monitor.RecentRequests.Count==40,"lifetime success/error/rejected counts are distinct and exceed the recent 40 cap");
 Check(first.Window.SampleCount==55&&first.Window.Mtp.AcceptanceRate==.6&&first.Window.Mtp.TokensPerRound==1.2&&first.Window.Cache.TokenHitRate==.25&&first.Window.Cache.RequestHitRate==1&&first.Window.Cache.Paths.Single().Requests==55,"request window aggregates weighted MTP and prefix reuse metrics");
 Check(first.Window.TtftP50Ms==250&&first.Window.TtftP95Ms==250&&first.Window.ProcP50Ms==200&&first.Window.PrefillP50Tps==150&&first.Window.DecodeP95Tps==50,"request percentiles use completed request timings and queue-adjusted TTFT");
 Check(first.Latest is {PrefillTps:null,DecodeTps:null,H2dBytes:null,DeviceKvTokens:4096},"first counter sample has no invented rate or transfer delta");
 monitor.ReadRequests(); var second=monitor.Sample(EngineStats(400,110,1024),gpu,monitorNow.AddSeconds(2));
 Check(second.Counts.Completed==55&&second.Latest is {PrefillTps:150,DecodeTps:50,H2dBytes:2048},"incremental read does not duplicate counts and rate uses actual elapsed seconds");
 monitor.UseEngine(run with {State="Stopped",Pid=null});
 Check(monitor.Sample(null,gpu,monitorNow.AddSeconds(4)).Counts.Completed==55,"stopping keeps current run counts and history");
 var resumed=monitor.Sample(EngineStats(500,150,2048),gpu,monitorNow.AddSeconds(6));
 Check(resumed.Latest is {PrefillTps:null,H2dBytes:null},"failed stats poll breaks throughput baseline instead of inventing zero samples");
 Check(monitor.Sample(EngineStats(1,1,0),gpu,monitorNow.AddSeconds(8)).Latest is {PrefillTps:null,DecodeTps:null,H2dBytes:null},"counter reset cannot produce negative rate or transfer values");
 var partialLine=RequestLine(58);
 File.AppendAllText(monitorLog,partialLine[..100]); monitor.ReadRequests();
 Check(monitor.Sample(null,gpu,monitorNow.AddSeconds(10)).Counts.Completed==55,"partial JSONL line waits for newline");
 File.AppendAllText(monitorLog,partialLine[100..]+"\nBROKEN JSON\n"); monitor.ReadRequests();
 Check(monitor.Sample(null,gpu,monitorNow.AddSeconds(12)).Counts is {Completed:56,MalformedLines:1},"completed partial line counts once and malformed line is reported");
 File.WriteAllText(monitorLog,RequestLine(59,timestamp:monitorNow.AddSeconds(1).ToUnixTimeMilliseconds())+"\n"); monitor.ReadRequests();
 Check(monitor.Sample(null,gpu,monitorNow.AddSeconds(14)).Counts.Completed==57,"same-file truncation restarts incremental read without resetting session counts");
 File.Move(monitorLog,monitorLog+".old");
 File.WriteAllText(monitorLog,RequestLine(59,timestamp:monitorNow.AddSeconds(1).ToUnixTimeMilliseconds())+"\n"+RequestLine(60,timestamp:monitorNow.AddSeconds(2).ToUnixTimeMilliseconds())+"\n"); monitor.ReadRequests();
 Check(monitor.Sample(null,gpu,monitorNow.AddSeconds(16)).Counts.Completed==58,"replaced log skips replayed records and counts the new request");
 var expired=monitor.Sample(null,gpu,monitorNow.AddSeconds(3610));
 Check(expired.Window.SampleCount==0&&expired.Window.Mtp.AcceptanceRate==null&&expired.Counts.Completed==58,"one-hour aggregate expires while lifetime counts remain");
 monitor.UseEngine(run with {RequestLogPath=Path.Combine(scratch,"another-run.jsonl"),StartedAt=monitorNow.AddHours(1)});
 var reset=monitor.Sample(EngineStats(),gpu,monitorNow.AddHours(1));
 Check(reset.Counts.Completed==0&&reset.History.Count==1&&reset.Latest!.PrefillTps==null,"new engine run resets requests, sample history and counter baseline");
}
File.WriteAllText(monitorLog,string.Join('\n',Enumerable.Range(1,4000).Select(id=>RequestLine(id)))+"\n");
using(var catchingUp=new MonitorSession()) {
 catchingUp.UseEngine(run); catchingUp.ReadRequests();var first=catchingUp.Sample(null,gpu,monitorNow);
 Check(first.Counts.CatchingUp&&first.Counts.UnreadBytes>0&&first.Counts.Completed<4000,"startup on a large existing log is bounded and advertises catch-up");
 for(var i=0;i<5;i++)catchingUp.ReadRequests();
 Check(catchingUp.Sample(null,gpu,monitorNow).Counts is {Completed:4000,CatchingUp:false},"large log catch-up completes without duplicating records");
}
using(var rotating=new MonitorSession()) {
 rotating.UseEngine(run);rotating.ReadRequests();
 File.Move(monitorLog,monitorLog+".rotated");
 File.WriteAllText(monitorLog,RequestLine(5000,timestamp:monitorNow.AddSeconds(1).ToUnixTimeMilliseconds())+"\n");
 for(var i=0;i<5;i++)rotating.ReadRequests();
 Check(rotating.Sample(null,gpu,monitorNow).Counts is {Completed:4001,CatchingUp:false},"rotation drains unread old-file records before consuming the replacement");
}
using(var limited=new MonitorSession()) {
 limited.UseEngine(run);
 for(var i=0;i<MonitorSession.RequestLimit+5;i++) {using var record=JsonDocument.Parse(RequestLine(i));limited.AddRequest(record.RootElement,monitorNow);}
 var limitedSample=limited.Sample(EngineStats(),gpu,monitorNow);
 Check(limitedSample.Window is {SampleCount:MonitorSession.RequestLimit,Truncated:true}&&limitedSample.Counts.Completed==MonitorSession.RequestLimit+5,"request window has an explicit memory cap independent of lifetime totals");
 if(Environment.GetEnvironmentVariable("NINFER_MONITOR_FIXTURE") is {Length:>0} fixtureOutput) {
  for(var i=1;i<=90;i++) limitedSample=limited.Sample(EngineStats(100+i*256,10+i*120,i*1048576),gpu with {UtilizationPercent=45+40*Math.Sin(i*.1),PowerWatts=160+60*Math.Sin(i*.1)},monitorNow.AddSeconds(i*2));
  File.WriteAllText(fixtureOutput,JsonSerializer.Serialize(new {engine=run,stats=EngineStats(),gpu,monitor=limitedSample,recentRequests=limited.RecentRequests,logTail="Fixture model loaded; no real GPU inference performed.",sampledAt=monitorNow},ConfigurationStore.Json));
 }
}
using(var historyMonitor=new MonitorSession()) {
 historyMonitor.UseEngine(run);
 MonitorSnapshot historySample=historyMonitor.Sample(EngineStats(),gpu,monitorNow);
 for(var i=1;i<=2000;i++)historySample=historyMonitor.Sample(EngineStats(i,i),gpu with {PowerWatts=i==1532?999:200},monitorNow.AddSeconds(i*2));
 Check(historySample.HistoryStoredSamples==2001&&historySample.HistoryDownsampled&&historySample.History.Count<=1800&&historySample.History.Any(point=>point.GpuPowerWatts==999),"six-hour raw history compacts API output while preserving observed power spikes");
 var expiredHistory=historyMonitor.Sample(null,gpu,monitorNow.AddHours(8));
 Check(expiredHistory.HistoryStoredSamples==1,"samples older than six hours expire from backend memory");
 var transferSpike=Enumerable.Range(0,2000).Select(i=>historySample.Latest! with {T=i,H2dBytes=i==511?999999:0,D2hBytes=i==532?555555:0,D2dBytes=i==561?333333:0,DeviceKvTokens=i==570?12345:0}).ToArray();
 var compacted=MonitorSession.CompactHistory(transferSpike);
 Check(compacted.Count<=1800&&compacted.Any(point=>point.H2dBytes==999999)&&compacted.Any(point=>point.D2hBytes==555555)&&compacted.Any(point=>point.D2dBytes==333333)&&compacted.Any(point=>point.DeviceKvTokens==12345),"API history compaction preserves each transfer and KV occupancy spike");
}
File.WriteAllText(monitorLog,RequestLine(100)+"\n");
using(var sampler=new TelemetryReader()) {
 sampler.Start(()=>run with {State="Stopped"},CancellationToken.None);
 await Task.Delay(2300);
 var sampled=await sampler.ReadAsync(run,CancellationToken.None);
 Check(sampled.Monitor.History.Count>=2&&sampled.Monitor.Counts.Completed==1,"background sampler collects history and log counts without browser requests");
 await sampler.StopAsync();
 await sampler.StopAsync();
 Check(true,"background sampler shutdown is idempotent");
}
var builder=WebApplication.CreateBuilder(); builder.WebHost.UseUrls("http://127.0.0.1:0"); builder.Logging.ClearProviders();
await using(var app=builder.Build()) {
 var fake=new FakeEngine(); int callbacks=0;
 ManagerApi.Map(app,store,fake,()=>{},()=>{callbacks++;if(!store.Settings.StartWithWindows)throw new InvalidOperationException("Registry fixture denied");});
 await app.StartAsync(); using var client=new HttpClient {BaseAddress=new Uri(app.Urls.Single())};
 using(var response=await client.PutAsJsonAsync("/api/profiles/xxs-160k",profile)) { if(!response.IsSuccessStatusCode)throw new Exception("Profile save API failed: "+response.StatusCode+" "+await response.Content.ReadAsStringAsync());Check(callbacks==0,"saving profile does not register Windows startup"); }
 using(var response=await client.PutAsJsonAsync("/api/settings",store.Settings with {StartWithWindows=false})) Check(response.StatusCode==System.Net.HttpStatusCode.Conflict&&callbacks==2&&store.Settings.StartWithWindows&&new ConfigurationStore(ManagerPaths.Select(scratch, ""),defaults).Settings.StartWithWindows,"failed registry update rolls settings back in memory and on disk");
 var settingsBeforeLanguage=store.Settings;var profileBeforeLanguage=JsonSerializer.Serialize(store.Profiles,ConfigurationStore.Json);
 using(var response=await client.PutAsJsonAsync("/api/language",new {language="en"})) Check(response.IsSuccessStatusCode&&callbacks==2&&store.Settings.Language=="en"&&new ConfigurationStore(ManagerPaths.Select(scratch, ""),defaults).Settings.Language=="en"&&JsonSerializer.Serialize(store.Settings with {Language=settingsBeforeLanguage.Language},ConfigurationStore.Json)==JsonSerializer.Serialize(settingsBeforeLanguage,ConfigurationStore.Json)&&JsonSerializer.Serialize(store.Profiles,ConfigurationStore.Json)==profileBeforeLanguage,"language saves independently without startup callback or inference parameter changes");
 using(var response=await client.PutAsJsonAsync("/api/language",new {language="fr"})) Check(response.StatusCode==System.Net.HttpStatusCode.BadRequest&&store.Settings.Language=="en"&&callbacks==2,"invalid language is rejected without settings mutation");
 using(var response=await client.PostAsync("/api/start/xxs-160k",null)) Check(response.StatusCode==System.Net.HttpStatusCode.Conflict,"synchronous start rejection is returned to caller");
 await app.StopAsync();
}
ThroughputWindowCheck.Run(Check);
await EngineTelemetryCheck.RunAsync(Check);
Console.WriteLine($"ALL {checks} CHECKS PASSED");
Console.WriteLine("Fixtures: "+scratch);

sealed class FakeEngine : IEngineController {
 public EngineSnapshot Snapshot=>new("Stopped"); public event Action? Changed {add{} remove{}}
 public Task StartAsync(LaunchSpec spec,CancellationToken cancellationToken=default)=>Task.FromException(new InvalidOperationException("Occupied fixture port"));
 public Task StopAsync(CancellationToken cancellationToken=default)=>Task.CompletedTask;
 public ValueTask DisposeAsync()=>ValueTask.CompletedTask;
}
