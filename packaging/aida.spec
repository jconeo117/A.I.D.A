# La receta del ejecutable (plan 78b). Modo carpeta: actualizar pisa la carpeta, y los
# datos del usuario viven fuera. Dos ejecutables sobre el mismo paquete: AIDA.exe sin
# consola para el icono, aida-cli.exe con consola para el CI y el diagnóstico.
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

RAIZ = Path(SPECPATH).parent
# Carpeta con tesseract.exe, sus DLL y tessdata/ (con spa). Solo al construir.
TESSERACT = os.environ.get("AIDA_TESSERACT_DIR")
if sys.platform == "win32" and not TESSERACT:
    raise SystemExit("falta AIDA_TESSERACT_DIR: sin Tesseract no se lee la historia")

datas = [
    (str(RAIZ / "packs"), "packs"),
    (str(RAIZ / "data" / "health" / "cie10"), "data/health/cie10"),
]
if TESSERACT:
    datas.append((TESSERACT, "tesseract"))
binaries = []
# Los brazos se abren por nombre desde arm.yaml (importlib): PyInstaller no los ve.
hiddenimports = collect_submodules("aida", filter=lambda n: not n.startswith("aida.relay"))
# Con DLL propias o importados dentro de funciones (onnxruntime, en el VAD).
for paquete in ("faster_whisper", "ctranslate2", "onnxruntime", "av",
                "pypdfium2", "pypdfium2_raw", "tokenizers"):
    d, b, h = collect_all(paquete)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    [str(RAIZ / "packaging" / "entrada.py")],
    pathex=[str(RAIZ / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
)
pyz = PYZ(a.pure)
gui = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AIDA", console=False, upx=False)
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name="aida-cli", console=True, upx=False)
COLLECT(gui, cli, a.binaries, a.datas, name="AIDA", upx=False)
