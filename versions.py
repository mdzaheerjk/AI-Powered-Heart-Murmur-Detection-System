import sys

def get_version(pkg):
    try:
        module = __import__(pkg)
        return module.__version__
    except Exception:
        return "Not Installed"

packages = [
"streamlit" ,
"tensorflow" ,
"numpy" ,
"pandas" ,
"scikit-learn" ,
"matplotlib" ,
"librosa" ,
"soundfile" ,
"huggingface-hub" 
]

print("\n📦 Installed Package Versions\n")
for pkg in packages:
    print(f"{pkg}: {get_version(pkg)}")

print("\nPython Version:", sys.version)
