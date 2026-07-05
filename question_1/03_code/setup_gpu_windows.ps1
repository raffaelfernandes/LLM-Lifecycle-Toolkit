param(
    [string]$VenvPath = ".\.venv-llm",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"

Write-Host "Criando ambiente virtual em $VenvPath"
python -m venv $VenvPath

$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

Write-Host "Atualizando pip"
& $PythonExe -m pip install --upgrade pip setuptools wheel

Write-Host "Instalando PyTorch com CUDA pelo indice: $TorchIndexUrl"
& $PythonExe -m pip install --upgrade torch torchvision torchaudio --index-url $TorchIndexUrl

Write-Host "Instalando dependencias de treino e leitura de documentos"
& $PythonExe -m pip install --upgrade transformers peft accelerate safetensors sentencepiece protobuf tqdm pymupdf pypdf

Write-Host "Validando CUDA no PyTorch"
@'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA nao ficou disponivel no PyTorch. Verifique driver NVIDIA e wheel CUDA.")
'@ | & $PythonExe -

Write-Host "Ambiente pronto. Ative com:"
Write-Host "$VenvPath\Scripts\Activate.ps1"
