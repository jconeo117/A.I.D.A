"""El relé: lo único de AIDA que vive en internet (102a).

Meta exige una dirección HTTPS pública para entregar lo que escribe un contacto, y el equipo
del cliente no la tiene ni debe abrir puertos. El relé recibe, comprueba la firma de Meta,
sella cada entrada con la clave pública del cliente y la guarda hasta que el equipo la recoge.
No puede leer lo que guarda y nunca envía nada.

Este paquete solo importa la biblioteca estándar, `cryptography` y `aida.relay.*`: se despliega
solo, sin el resto de AIDA. Lo vigila `tests/test_relay_isolation.py`.
"""
