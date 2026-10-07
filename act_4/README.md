# MUTLF: Mega_Ultra_Turbo_Lite_Fastapi (example)

*Esta es una actividad del curso de licenciatura Cloud Computing, 2026.*

#### Indicaciones:
1. Revisa el código.
2. Implementa un servicio similar. Sin embargo, en esta ocasión deberás realizar análisis de sentimientos sobre datos financieros, utilizando esta base de datos:

```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("sbhatti/financial-sentiment-analysis")

print("Path to dataset files:", path)

import pandas as pd

data = pd.read_csv(path+"\data.csv")
data.head()
```

Donde el y-value es la columna "sentiment" y el x-value es la columna "sentence". Utilicen un transformer pequeñito para que sea agnóstico al vocabulario.

3. Utilicen un entorno virtual.
4. El entregable es su propio repositorio.


### Entregas:
- Por equipos.
- La siguiente semana.
- :)


### FAQ's:

* ¿Esta actividad usa Azure?

Esta actividad no usa Azure, todo es localmente. La semana que entra veremos a detalle este salto.

* ¿Cómo garantizo agnosticismo al vocabulario?

Los transformers ya vienen con un tokenizador basado en byte-pairing. Utilicen un modelillo pequeño tipo BERT.

* ¿La mayonesa es un estimador estadístico?
  
Ci.



⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣴⠶⣿⣛⣛⣋⠉⠛⠛⠛⠶⢶⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣾⠟⠛⠁⠀⠉⠉⠉⠙⠻⢿⣿⣶⠶⣦⣌⡙⠿⣶⣄⡀⠀⣀⣤⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⡿⢿⠿⣛⣿⣟⣻⢶⣤⣄⠀⠀⠀⠉⠿⣦⡀⠉⠻⢷⣾⣽⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣠⣾⣿⣿⣤⣼⣿⣿⣿⣿⣿⣷⣾⣿⣷⣄⠀⠀⠀⢻⣷⣴⣶⣾⣿⣿⣏⣿⣿⣿⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣼⣿⣿⣿⡿⢽⣿⣿⣿⣿⣿⣿⣿⣭⣭⡏⢻⡆⠀⠀⢸⣿⣿⣿⠛⠉⠙⣿⣿⣿⣿⣿⣿⣦⣴⠿⠛⠛⠿⣦⣄⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢸⣿⣿⣿⣿⡇⠈⠻⣿⣿⣿⣿⣿⢿⣿⡏⠁⢸⣿⡇⠀⢸⣿⣿⡋⠀⠀⣰⣿⣿⣿⠿⣿⣿⣿⣿⣶⣶⣄⠁⠀⠙⢿⣦⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⠆⠀⠿⠛⠋⠉⠁⠻⠋⠁⠀⣾⣧⠀⠀⢸⣿⣿⣥⣤⣾⣿⣿⡿⠃⣰⣿⣿⣿⣿⣿⡛⢿⣷⣄⠀⠈⠻⣧⠀⠀⠀⠀
⠀⠀⠀⠀⠈⣻⣿⣿⣿⣿⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣿⡟⠀⠀⢸⣿⣿⣿⣿⣿⣿⡿⢠⣾⣿⠏⠀⣿⣿⣿⣿⣎⡹⣿⡄⠀⠀⢹⣧⠀⠀⠀
⠀⠀⠀⠀⠀⠉⣿⠋⢻⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⣠⣿⣿⡇⠀⠀⣿⡿⠏⣿⣿⣿⡀⠐⠿⠟⠀⠀⢰⡟⣿⣿⣿⣿⣆⠸⣿⡄⠀⠀⢿⡆⠀⠀
⠀⠀⠀⠀⠀⢸⣿⠁⠈⠛⠛⡀⠀⠀⠀⠀⠀⠀⠸⠿⠟⠸⣧⡀⢰⡿⠀⢰⣿⣿⠘⣿⡄⠀⠀⠀⠀⣿⣿⣿⠛⢿⡟⢿⡄⠻⣧⠀⠀⢸⡇⠀⠀
⠀⠀⠀⠀⢀⣼⣿⠀⠀⠀⣼⣿⠛⠻⠶⢤⣤⣄⡀⠀⠀⠀⠹⣧⣸⣇⠀⢸⣿⣿⣇⡌⢿⡆⠀⠀⣸⣿⢀⣿⠀⢸⣇⠘⠃⡄⣿⠀⠀⠈⣿⠀⠀
⠀⠀⠀⠀⢼⣿⣿⣧⠀⢠⣿⠋⠀⠀⠀⠀⠈⠉⠛⠿⣦⣄⣀⠙⠻⣿⣆⣸⣿⣿⣿⡇⠘⣿⡀⢀⣿⠁⢸⡿⠆⠀⣿⡀⢠⣿⡿⡇⠀⠈⣿⠀⠀
⠀⠀⠀⠀⠀⢠⣿⣿⡄⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠿⣧⣄⣈⠙⠻⣿⣿⡿⠇⠀⢻⣿⣿⡏⠀⣼⠇⠀⠀⣿⡇⣼⣿⣟⠀⠀⠀⣿⠃⠀
⠀⠀⠀⠀⠀⢾⡿⢻⣷⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⢿⣤⡀⠙⠛⠁⠀⠀⠘⣿⡏⠀⢠⣿⠀⠀⠀⣿⣱⣿⣿⠉⠀⠀⢀⣿⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣷⣄⣠⡴⠿⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⢷⣤⣀⠀⠀⠀⣿⡇⢀⣾⠇⠀⠀⢰⣿⣟⣿⠇⠀⠀⠀⣿⡿⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢸⣿⠹⣿⣿⠏⠀⠀⠙⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠛⠳⠛⣹⣿⣾⠏⠀⠀⠀⣸⣯⣾⠃⠀⠀⠀⢠⣿⣿⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢸⡏⠀⢸⣿⣧⠀⠀⠀⠘⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣿⡿⠋⠀⠀⠀⣰⣿⡿⠃⠀⠀⠀⠀⣼⣿⡇⠀⠀
⠀⠀⠀⠀⠀⠀⠀⣿⠇⠀⣾⠸⣿⠀⠀⠀⠀⢹⡇⠀⠀⢀⣀⣀⣀⣀⣀⠀⠀⠀⠀⠀⠀⣿⠁⠀⠀⠀⠀⣰⣿⡿⠁⠀⠀⠀⠀⣰⣿⣿⠁⠀⠀
⠀⠀⠀⠀⠀⠀⢰⡿⠇⢀⣿⠀⣿⣀⣠⣴⣶⣾⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣿⣄⠀⠀⠀⣼⣿⠟⠀⠀⠀⠀⢀⣴⣿⣿⣿⠀⠀⠀
⠀⠀⠀⠀⠀⢠⣿⣇⠀⢸⣇⣾⣿⣿⣿⣿⣿⣿⣿⣿⠶⠶⠷⠶⠿⠷⠷⠾⠿⠿⢿⣿⣿⣿⣿⠀⢀⣾⣿⠋⠀⠀⠀⠀⢀⣾⣿⣿⣿⡇⠀⠀⠀
⠀⠀⠀⠀⢠⣿⠋⠀⠀⣾⠁⣿⡿⠓⠀⢻⣿⣿⣿⣿⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⠀⢸⣿⣠⣿⡿⠃⠀⠀⠀⠀⢀⣾⣿⡟⢸⣿⡇⠀⠀⠀
⠀⠀⠀⢠⣿⠃⠀⠀⢰⡟⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣭⣽⣿⣿⣻⣿⣿⣿⣿⠁⠀⠀⠀⠀⠀⣼⣿⡿⠁⣸⣿⠃⠀⠀⠀
⠀⠀⠀⣿⠃⣼⠇⠀⣾⠃⣻⡿⠛⠉⠉⠁⢹⣿⣿⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⢉⣿⣿⡇⠀⠀⠀⠀⠀⣸⣯⣿⠃⢀⣿⣿⠀⠀⠀⠀
⠀⠀⣸⡯⣼⡏⠀⣸⡟⠀⣿⣧⠀⠀⠀⠀⣿⣿⣿⣿⡿⣿⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⡟⠀⠀⠀⠀
⢀⣴⣿⣷⣿⠁⣰⣿⣀⣰⣿⡅⠀⠀⢀⣼⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣻⣷⣦⣄⡀
