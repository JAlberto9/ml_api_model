# ML CHALLENGE

Repositorio de entrenamiento de modelo de ML y disponibilizado en aun API con respaldo en un DB para cada request a la API.


## Prerrequisitos

1. Docker
2. Python

## Tecnologías

* Python
* FastAPI
* PostgreSql

## Estructura del proyecto

```
.
├── alembic/                        # Codigo de la DB para migraciones  
├── api/                          
│   ├── exceptions.py              # Manejo de los erroes al llamado de la API 
│   └── feature_transformer.py     # Procesamiento del body del request para usar el modelo
│   └── main.py                    # Archivo principal con los paths del API
│   └── models.py                  # Modelo de datos para los request
│   └── schema.py                  # Declaracion para DB
└── src/    
│   └── data/                      # Directorio para colocar la data raw para el train del modelo y para el predict local
│   └── metrics/
        └── scores.json            # Resultados del entrenamiento del modelo.
│   └── model/
        └── MODEL_CLASSIFIEr.xgb   # Modelo obtenido del ultimo entrenamiento
└── predict/    
│   └── model.py                   # Archivo para cargar el modelo
│   └── predict.py                 # Calcula predicicones de manera local
└── train/    
│   └── data.py                    # Declaracion de las features para el modelo
│   └── train.py                   # Codigo usado para el entreneminto del modelo
├── env-example                    # Ejemplo de las variables de entorno para poder probar el modelo
├── requirements.txt               # Python dependencias para el MS
├── Dockerfile                     # Archivo Dockerfile para construir la imagen
├── docker-compose.yml             # yml con la definicion de cada servicio para levantar el MS

```



## Levantando MS 🚀
En caso de nover las versiones de la difinicion de la db en alembic corre los siquientes comandos

Generamos la migracion para la definicion de la DB
```
docker-compose run app alembic revision --autogenerate -m "New Migration"
```
```
docker-compose run app alembic upgrade head
```

Ahora contruye y levanta el docker compose

```
docker-compose build
docker-compose up
```

