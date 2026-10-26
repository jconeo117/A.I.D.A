# AIDA: guía del operador

Para quien construye, firma y entrega AIDA. No es para el cliente: esa es `GUIA-CLIENTE.md`.

## 1. Publicar una versión

1. Suba la versión en `pyproject.toml` y en `src/aida/__init__.py`. Deben coincidir: el CI
   lo comprueba y falla si no.
2. `uv lock`, commit, y etiqueta: `git tag vX.Y.Z`.
3. `git push` y `git push --tags`.
4. Baje el instalador desde **Actions** o **Releases** del repositorio.
5. Revise el artefacto `bateria-windows`: la batería de Windows no bloquea todavía, y cada
   fallo se registra como issue.

## 2. Construir sin etiqueta

En **Actions**, abra el flujo `windows` y pulse **Run workflow**. Sale un `AIDA-Setup-X.Y.Z.exe`
como artefacto, sin publicarse.

## 3. Firmar una licencia

Desde la raíz del repositorio, con la clave privada fuera del repositorio:

```bash
uv run python tools/licencias.py sign ~/Dev/AIDA-Licences/aida-licencias.key \
  --client "Nombre del cliente" --arms health --expires AAAA-MM-DD --out licencia.json
uv run python tools/licencias.py show licencia.json
```

`show` comprueba la firma y muestra qué da la licencia. La clave privada no se entrega
nunca, ni se sube al repositorio.

## 4. Qué se entrega

- El `Setup` (`AIDA-Setup-X.Y.Z.exe`).
- La `licencia.json`.
- La `GUIA-CLIENTE.md`.
- La clave de OpenAI, si se usa la nube: de un proyecto aparte, con límite de gasto.

## 5. Dónde va la licencia en el equipo del cliente

AIDA lee la licencia de `%APPDATA%\AIDA\licencia.json`. Hoy no hay pantalla en la aplicación
que la cargue cuando la licencia falta: el operador la copia en esa carpeta, con AIDA
cerrada. Después, al abrir AIDA, la licencia se verifica sola.

## 6. Diagnóstico en el equipo del cliente

Con AIDA instalada, desde PowerShell:

```powershell
& "$env:LOCALAPPDATA\Programs\AIDA\aida-cli.exe" --version
& "$env:LOCALAPPDATA\Programs\AIDA\aida-cli.exe" --check-capture
& "$env:LOCALAPPDATA\Programs\AIDA\aida-cli.exe" --audit
```

`--check-capture` muestra una línea por módulo, el modelo y Tesseract. Una línea `FALTA`
dice qué módulo no carga.

## 7. Lo que todavía no hace

- Firma del ejecutable: sin firmar, Windows avisa al instalar.
- Actualización automática: el cliente instala el `Setup` nuevo a mano.
- Registro de errores: el diagnóstico va a un nulo, a propósito, para no dejar texto
  clínico en un archivo.
- Icono propio: lleva el de PyInstaller.
