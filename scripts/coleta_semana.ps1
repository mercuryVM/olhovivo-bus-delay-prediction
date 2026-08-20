# Supervisor da coleta de 1 semana.
#
# Mantem o coletor vivo: se o processo morrer (queda de rede, erro nao
# tratado, notebook que dormiu), reinicia sozinho. A janela de 7 dias NAO
# reinicia junto — ela fica gravada em dados/estado_coleta.json, entao o
# supervisor sempre continua a mesma semana.
#
#   powershell -ExecutionPolicy Bypass -File scripts\coleta_semana.ps1
#
# Parametros:
#   -Horas        duracao total (padrao: le do config/coleta.yaml)
#   -Reiniciar    comeca uma janela nova, descartando a anterior
#   -EsperaS      segundos entre uma queda e a proxima tentativa

param(
    [double] $Horas = 0,
    [switch] $Reiniciar,
    [int]    $EsperaS = 30
)

$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $PSScriptRoot
Set-Location $raiz

$python = Join-Path $raiz ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$logDir = Join-Path $raiz "dados\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "supervisor.log"

function Escrever($texto) {
    $linha = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $texto
    Write-Host $linha
    Add-Content -Path $log -Value $linha -Encoding utf8
}

$argumentos = @("-m", "olhovivo", "coletar")
if ($Horas -gt 0)  { $argumentos += @("--horas", $Horas) }

Escrever "supervisor iniciado (python: $python)"

# --reiniciar so vale na PRIMEIRA tentativa; nas seguintes seria destrutivo,
# porque zeraria a janela da semana a cada queda.
$primeira = $true
$tentativa = 0

while ($true) {
    $tentativa++
    $atual = $argumentos
    if ($primeira -and $Reiniciar) { $atual = $argumentos + "--reiniciar" }
    $primeira = $false

    Escrever "tentativa $tentativa -> $python $($atual -join ' ')"
    & $python $atual
    $codigo = $LASTEXITCODE

    if ($codigo -eq 0) {
        Escrever "coleta concluiu normalmente (exit 0). Encerrando supervisor."
        break
    }
    if ($codigo -eq 130) {
        Escrever "interrompido pelo usuario (Ctrl+C). Encerrando supervisor."
        break
    }
    if ($codigo -eq 2) {
        Escrever "erro de configuracao (exit 2) — reiniciar nao adianta. Encerrando."
        break
    }

    Escrever "coletor caiu com exit $codigo; nova tentativa em $EsperaS s"
    Start-Sleep -Seconds $EsperaS
}

Escrever "== status final =="
& $python -m olhovivo status
