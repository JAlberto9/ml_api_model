# ML API MODEL ARCH

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

##  Docs 🚀
FastAPI tiene integrado la doc para poder ver los endpoints de nuestro MS, para poder acceder solo entra en esta direccion

```
http://localhost:8000/docs
```

Para ingresar a pgAdmin y verificar los registros ingestados acceder al UI

```
http://localhost:5050/browser/
```
Para este ultimo recuerda loguearte con los datos aue colocaste en el .env

Estas listo para hacer las pruebas al MS puedes usar la misma interfaz de FastAPI, POSTMAN, con el que te siguentas mas agusto.

Agrego un ejemplo de body que puede recibir 

```
curl -X 'POST' \
  'http://localhost:8000/calculate/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
   "data":[
      {
         "order_id":123,
         "store_id":34,
         "to_user_distance":643,
         "to_user_elevation":223,
         "total_earning":53
      },
      {
         "order_id":345,
         "store_id":123,
         "to_user_distance":123,
         "to_user_elevation":3123,
         "total_earning":1232
      },
      {
         "order_id":5667,
         "store_id":56,
         "to_user_distance":789,
         "to_user_elevation":32,
         "total_earning":34
      }
   ]
}'
```

# TRAIN MODEL
Ahora te explicaremos como entrenar tu modelo tu modelo

## Requisitos

Indispensable para esta parte usar tu entornovirtual, si tienes MacOS lo pudes inizilicar de la siguiente forma:
```
virtualenv venv
```
Accedemos al entorno virtual
```
source venv/bin/activate
```
Despues de eso estaras dentro del entorno virtual, procedemos a instalar los requirements del repositorio

```
pip install -r requirements.txt
```
## Entrenamiento

Dentro del codigo de train.py tenemos todo el pipeline de entrenamiento, desde la busqueda de hierparmetros hasta el print de las distintas metricas que sacamos durante el entrenmiento.
```
 python src/train/train.py 
```
## Predict

Podemos calcular el target para csv en bach, colocando la data en el directorio de data y solo corriendo el siguiente script podemos evaluar a todo el csv con el modelo.
```
 python src/train/train.py 
```
