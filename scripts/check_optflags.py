import sys
import sys
import json
import os


def show(path, key, label):
    print("===== %s" % label)
    print("  cache: %s" % path)
    cache = os.path.join(os.path.dirname(os.path.join(path, "x")), "CMakeCache.txt")
    if os.path.isfile(cache):
        with open(cache, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.startswith(("CMAKE_BUILD_TYPE:", "CMAKE_CXX_FLAGS_RELEASE:")):
                    print("  " + line.strip())
    else:
        print("  CMakeCache.txt not found at %s" % cache)

    cc = os.path.join(path, "compile_commands.json")
    if not os.path.isfile(cc):
        print("  compile_commands.json missing")
        return
    with open(cc, encoding="utf-8") as fh:
        entries = json.load(fh)
    for e in entries:
        f = e["file"].replace("\\", "/")
        if f.endswith(key):
            cmd = e.get("command") or " ".join(e.get("arguments", []))
            print("  has -O3     : %s" % (" -O3" in cmd))
            print("  has -DNDEBUG: %s" % ("-DNDEBUG" in cmd))
            print("  full command follows")
            print(cmd)
            return
    print("  %s not found in database" % key)


if len(sys.argv) == 3:
    show(sys.argv[1], sys.argv[2], "BUILD")
else:
    print(__doc__)
