# AIDA: guía para empezar

AIDA es un asistente para la consulta. Escucha o lee la historia, propone la nota y los
códigos, y usted decide qué se guarda. Todo se queda en este equipo, salvo lo que usted
envíe a la nube.

## 1. Lo que necesita

- Windows 10 u 11 de 64 bits.
- Unos 1,5 GB libres en el disco: el programa, unos 500 MB sin contar el lector de documentos
  (medido en la construcción de prueba), y el modelo de voz, unos 500 MB.
- Chrome o Edge.
- Un micrófono.
- Internet, para la nube y para descargar el modelo la primera vez.

## 2. Instalar

1. Abra el archivo `AIDA-Setup-…exe`.
2. Windows puede mostrar «Windows protegió su PC». Pulse **Más información** y después
   **Ejecutar de todas formas**. Aparece porque AIDA todavía no está firmado digitalmente.
3. No pide contraseña de administrador. Siga las pantallas y pulse **Instalar**.
4. Al terminar se abre AIDA. Después, el icono **AIDA** del escritorio y del menú Inicio
   lo abre de nuevo.

## 3. Primer arranque

1. AIDA muestra **Primer arranque**. Escriba su nombre tal como aparece en la historia, su
   registro profesional y un PIN de 4 a 8 cifras, y repítalo.
2. AIDA le muestra un **código de recuperación**. Guárdelo fuera de este equipo: sirve para
   cambiar el PIN si lo olvida.
3. Entra en AIDA. Si el equipo aún no tiene la licencia, el programa abre la agenda y
   no permite nada más hasta que la licencia esté en su sitio (ver la guía del operador).

Para una copia de seguridad, en la misma primera pantalla está **¿Viene de otro equipo?
Restaurar una copia**. Si ya tiene una copia guardada, puede restaurarla aquí.

## 4. Uso diario

- Abra AIDA con su icono y escriba su PIN.
- Termine con **Salir** cuando acabe.
- Si cierra la pestaña sin **Salir**, AIDA sigue abierta en segundo plano. Pulse el icono
  otra vez: vuelve a la misma sesión y no se abre una segunda copia.
- Si se queda sin usar un rato, AIDA le pide el PIN otra vez.

## 5. Actualizar

Instale el `Setup` nuevo encima del anterior. Sus perfiles, su licencia y sus consultas
siguen en su sitio.

## 6. Desinstalar

Desinstale AIDA desde **Aplicaciones instaladas** de Windows. Sus datos no se borran: quedan
en `%APPDATA%\AIDA` y `%LOCALAPPDATA%\AIDA`. Para borrarlos del todo, hágalo a mano, y solo
si ya no los necesita.

## 7. Copia de sus datos

- La configuración (perfiles, claves, licencia) se guarda con **Ajustes → Copia de
  seguridad**. Pida una frase de 12 caracteres o más: sin ella la copia no sirve, y AIDA no
  la puede recuperar. Guarde la copia y la frase en sitios distintos, fuera de este equipo.
- Las consultas están cifradas en `%LOCALAPPDATA%\AIDA`, y solo las lee AIDA en este equipo.
  Para conservarlas en otro sitio, copie esa carpeta, y guarde también la copia de la
  configuración.
