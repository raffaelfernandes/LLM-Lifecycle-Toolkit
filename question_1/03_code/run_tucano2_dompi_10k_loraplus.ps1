param(
    [string]$VenvPath = ".\.venv-llm",
    [string]$ModelId = "Polygl0t/Tucano2-qwen-0.5B-Base",
    [string]$DataDir = "llm_pretraining\data_dompi_2025_tucano2_10k",
    [string]$OutputDir = "llm_pretraining\runs\tucano2_qwen_0p5b_loraplus_dompi_10k",
    [int]$Epochs = 3,
    [int]$MaxSteps = 0,
    [int]$BlockSize = 512,
    [int]$GradAccumSteps = 16,
    [int]$EvalEvery = 500,
    [int]$SaveEvery = 1000,
    [int]$EvalMaxBatches = 96,
    [int]$BenchmarkMaxItems = 25,
    [switch]$Resume,
    [switch]$RebuildTokenCache,
    [switch]$SkipPrepare,
    [switch]$PrepareOnly
)

$ErrorActionPreference = "Stop"

$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

if (-not $SkipPrepare) {
    Write-Host "Preparando experimento DOMPI/Tucano2 10k sem vazamento..."
    & $PythonExe "llm_pretraining\prepare_tucano2_dompi_10k_experiment.py" `
        --output-dir $DataDir `
        --train-docs 10000 `
        --valid-docs 1000 `
        --test-docs 1000 `
        --benchmark-size 25 `
        --context-chars 2400
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($PrepareOnly) {
    Write-Host "Preparo concluido. Treino nao iniciado porque -PrepareOnly foi usado."
    exit 0
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "GPU detectada:"
    nvidia-smi
}

$ArgsTreino = @(
    "llm_pretraining\train_continued_pretraining_gpu.py",
    "--model-id", $ModelId,
    "--data-dir", $DataDir,
    "--output-dir", $OutputDir,
    "--epochs", "$Epochs",
    "--block-size", "$BlockSize",
    "--stride", "$BlockSize",
    "--train-batch-size", "1",
    "--eval-batch-size", "1",
    "--grad-accum-steps", "$GradAccumSteps",
    "--learning-rate", "0.0001",
    "--warmup-steps", "100",
    "--eval-every", "$EvalEvery",
    "--save-every", "$SaveEvery",
    "--eval-max-batches", "$EvalMaxBatches",
    "--benchmark-max-items", "$BenchmarkMaxItems",
    "--benchmark-max-new-tokens", "128",
    "--use-loraplus",
    "--loraplus-lr-ratio", "16"
)

if ($MaxSteps -gt 0) {
    $ArgsTreino += @("--max-steps", "$MaxSteps")
}

if ($Resume) {
    $ArgsTreino += @("--resume-from", "latest")
}

if ($RebuildTokenCache) {
    $ArgsTreino += "--rebuild-token-cache"
}

Write-Host "Iniciando pre-treino continuado Tucano2 + LoRA+..."
& $PythonExe @ArgsTreino
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
