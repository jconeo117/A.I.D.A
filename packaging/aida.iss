; El instalador (plan 78b). Por usuario y sin administrador: en el equipo del médico no hace
; falta el técnico. Los datos viven en %APPDATA%\AIDA y %LOCALAPPDATA%\AIDA: ni actualizar
; ni desinstalar los tocan.
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{D2815DD6-CE2D-4369-A248-55C79F46935B}
AppName=AIDA
AppVersion={#AppVersion}
AppPublisher=Jhon Coneo
DefaultDirName={autopf}\AIDA
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=AIDA-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
CloseApplications=yes
RestartApplications=no
UninstallDisplayIcon={app}\AIDA.exe
WizardStyle=modern

[Languages]
Name: "es"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; Una actualización no deja DLL de la versión anterior.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\dist\AIDA\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\AIDA"; Filename: "{app}\AIDA.exe"
Name: "{autodesktop}\AIDA"; Filename: "{app}\AIDA.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\AIDA.exe"; Description: "{cm:LaunchProgram,AIDA}"; Flags: nowait postinstall skipifsilent
