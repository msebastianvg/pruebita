import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
# import seaborn as sns
# from sklearn.linear_model import LinearRegression

st.title('Reporte BETS - 2023')
st.subheader('Comienza la fiesta damas y caballeros.')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data





file_path = 'bets - gh.xlsx'
#file_path = 'https://github.com/msebastianvg/pruebita/blob/726e5dc6981883405d77349748318ec156c82583/bets%20-%20gh.xlsx'  # Replace with the actual path to your Excel file
df = pd.read_excel(file_path)
# st.write(df.head())

total_wins = (df['WL'] == 1).sum()
total_losses = (df[df['WL'] == 0]['WL'] == 0).sum()
#st.subheader('W: ' + str(total_wins))
#st.subheader('L: ' + str(total_losses))
media = total_wins/(total_losses+total_wins)
#st.subheader('Win rate: ' + str("{:.1f}%".format(media * 100)))


apuestas_ganadas = total_wins
apuestas_perdidas = total_losses

# Crear un gráfico de barras interactivo con Plotly Express
fig = px.bar(
    x=["Ganadas", "Perdidas"],
    y=[total_wins, total_losses],
    color=["Ganadas", "Perdidas"],
    labels={"x": "Tipo de Apuesta", "y": "Cantidad"},
    title="Apuestas Ganadas vs. Apuestas Perdidas",
)

# Mostrar el gráfico interactivo en Streamlit usando st.write
st.write(fig)



# df = pd.read_excel('E:\\B\\bets.xlsx')

# Convertir la columna 'date' a tipo datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Filtrar las apuestas ganadas (WL=1)
apuestas_ganadas = df[df['WL'] == 1]

# Agrupar por fecha y contar la cantidad de apuestas ganadas por día
ganancias_por_dia = apuestas_ganadas.groupby('DATE')['WL'].count()

# Crear un gráfico de barras
st.write("Cantidad de Apuestas Ganadas por Día")
st.bar_chart(ganancias_por_dia)





# Cargar los datos desde el archivo Excel
# df = pd.read_excel('E:\\B\\bets.xlsx')

# Calcular el valor mínimo y máximo de la columna 'PERCENTAGE'
min_single = df['PERCENTAGE'].min()
max_single = df['PERCENTAGE'].max() + 0.01

# Convertir la columna 'date' a tipo datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Seleccionar el último registro de cada fecha
df = df.groupby('DATE').tail(1)

# Calcular el promedio global de 'PERCENTAGE'
average_single_global = df['PERCENTAGE'].mean()

# Crear un gráfico de barras que muestra la variación de 'PERCENTAGE' con respecto a las fechas
st.write("Porcentaje de ganancias")

fig, ax = plt.subplots(figsize=(10, 6))

# Asignar colores según 'WL' en el último registro de cada fecha
colors = ['lightgreen' if wl == 1 else 'orange' for wl in df['WL']]

ax.bar(df['DATE'], df['PERCENTAGE'], color=colors)

# Personalizar el gráfico
ax.set_xlabel('FECHA')
ax.set_ylabel('GANANCIAS (%)')

# Establecer el formato de los valores en el eje Y como porcentajes con un decimal
ax.yaxis.set_major_formatter('{:.0%}'.format)

ax.set_title('GANANCIAS (%) POR FECHA')
ax.set_ylim([min_single, max_single])  # Establecer el límite superior del eje Y

# Rotar las etiquetas del eje x para una mejor legibilidad
plt.xticks(rotation=45)

# Agregar la línea que muestra el promedio global de 'PERCENTAGE'
ax.axhline(average_single_global, color='blue', linestyle='--', label='PROMEDIO DE GANANCIAS (%)')

# Mostrar la leyenda
ax.legend()

# Mostrar el gráfico en Streamlit
st.pyplot(fig)



# df = pd.read_excel('E:\\B\\bets.xlsx')

# Calcular el valor mínimo de la columna 'SINGLE'
min_single = df['PERCENTAGE'].min()

# Calcular el valor máximo de la columna 'SINGLE' y agregar 1000
max_single = df['PERCENTAGE'].max() + 0.01

# Convertir la columna 'date' a tipo datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Seleccionar el último registro de cada fecha
df = df.groupby('DATE').tail(1)

# Calcular el promedio global de 'SINGLE'
average_single_global = df['PERCENTAGE'].mean()

# Crear un gráfico de barras que muestra la variación de 'SINGLE' con respecto a las fechas
st.write("Variación de 'SINGLE' con respecto a las Fechas")

fig, ax = plt.subplots(figsize=(10, 6))

# Asignar colores según 'WL' en el último registro de cada fecha
colors = ['lightgreen' if wl == 1 else 'pink' for wl in df['WL']]

ax.bar(df['DATE'], df['PERCENTAGE'], color=colors)

# Personalizar el gráfico
ax.set_xlabel('FECHA')
ax.set_ylabel('POZO TOTAL')
ax.set_title('DETALLE DEL POZO')
ax.set_ylim([min_single, max_single])  # Establecer el límite superior del eje Y
ax.yaxis.set_major_formatter('{:.0%}'.format)

# Rotar las etiquetas del eje x para una mejor legibilidad
plt.xticks(rotation=45)

# Ajustar una curva cuadrática (segundo grado) a los datos
x = np.arange(len(df['DATE']))
y = df['PERCENTAGE'].values
curve_fit = np.polyfit(x, y, 2)
quadratic_curve = np.poly1d(curve_fit)

# Calcular valores de la curva cuadrática
quadratic_values = quadratic_curve(x)

# Agregar la línea que muestra el promedio global de 'SINGLE'
ax.axhline(average_single_global, color='blue', linestyle='--', label='PROMEDIO que no dice nada')

# Agregar la curva cuadrática al gráfico
ax.plot(df['DATE'], quadratic_values, label='CURVA que tengo que recalcular', color='purple')

# Mostrar la leyenda
ax.legend()

# Mostrar el gráfico en Streamlit
st.pyplot(fig)



# df = pd.read_excel('E:\\B\\bets.xlsx')

reference_date = pd.to_datetime("2023-01-01")  # Establece tu fecha de referencia
df['DAYS_FROM_REFERENCE'] = (df['DATE'] - reference_date).dt.days

# Extrae los datos de 'DAYS_FROM_REFERENCE' y 'SINGLE' del DataFrame
X = df['DAYS_FROM_REFERENCE'].values
Y = df['PERCENTAGE'].values

# Calcula la media de X e Y
mean_X = np.mean(X)
mean_Y = np.mean(Y)

# Calcula la desviación estándar de X e Y
std_X = np.std(X)
std_Y = np.std(Y)

# Calcula el coeficiente de la pendiente (a) y el intercepto (b) de la regresión lineal
a = np.sum((X - mean_X) * (Y - mean_Y)) / np.sum((X - mean_X) ** 2)
b = mean_Y - a * mean_X

# Construye la ecuación lineal Y = aX + b
equation = f"Y = {a:.2f}X + {b:.2f}"

# Imprime la ecuación lineal
print("Ecuación Lineal:")
print(equation)





# df = pd.read_excel('E:\\B\\bets.xlsx')

reference_date = datetime(2023, 1, 1)  # Establece tu fecha de referencia
df['DAYS_FROM_REFERENCE'] = (df['DATE'] - reference_date).dt.days

# Extrae los datos de 'DAYS_FROM_REFERENCE' y 'SINGLE' del DataFrame
X = df['DAYS_FROM_REFERENCE'].values
Y = df['PERCENTAGE'].values

# Calcula la media de X e Y
mean_X = np.mean(X)
mean_Y = np.mean(Y)

# Calcula la desviación estándar de X e Y
std_X = np.std(X)
std_Y = np.std(Y)

# Calcula el coeficiente de la pendiente (a) y el intercepto (b) de la regresión lineal
a = np.sum((X - mean_X) * (Y - mean_Y)) / np.sum((X - mean_X) ** 2)
b = mean_Y - a * mean_X

# Construye la ecuación lineal Y = aX + b
equation = f"Y = {a:.2f}X + {b:.2f}"

# Imprime la ecuación lineal
print("Ecuación Lineal:")
print(equation)


a = 554.83
b = 518903.93

df['DATE'] = pd.to_datetime(df['DATE'])

# Ordena el DataFrame por la columna 'DATE'
df.sort_values(by='DATE', inplace=True)

# Extrae los datos de 'DATE' y 'SINGLE' del DataFrame
X = df['DATE'].values
Y = df['PERCENTAGE'].values

# Convierte las fechas en días transcurridos desde una fecha de referencia
reference_date = X[0]  # Establece la fecha de referencia como la primera fecha en tus datos
X = (X - reference_date).astype('timedelta64[D]').astype(int)

# Cálculo de la función lineal
def linear_regression(x):
    a = 0.001
    b = -0.007
    #b = 518903.93
    #a = (Y[-1] - Y[0]) / (X[-1] - X[0])  # Calcula la pendiente 'a' usando el primer y último punto
    #b = Y[0] - a * X[0]  # Calcula el término de intercepción 'b'
    return a * x + b

# Crea una gráfica
fig, ax = plt.subplots()
ax.plot(X, Y, label='GANANCIA (%)', linestyle='-', marker='o')
ax.plot(X, linear_regression(X), label='REGRESIÓN LINEAL', linestyle='--', color='red')
ax.set_xlabel('FECHA')
ax.set_ylabel('GANANCIA (%)')
ax.set_title('REGRESIÓN LINEAL que está de locos')
plt.xticks(rotation=45)  # Rotación de las etiquetas del eje X para mejor visualización
plt.legend()
ax.yaxis.set_major_formatter('{:.0%}'.format)

# Muestra la gráfica en Streamlit
st.pyplot(fig)

media_single = df['PERCENTAGE'].mean()

# Calcula el error medio absoluto (MAE)
mae = (df['PERCENTAGE'] - media_single).abs().mean()

print(f'Error Medio Absoluto (MAE): {mae:.2f}')

desviacion_estandar = df['PERCENTAGE'].std()

print(f'Desviación Estándar de SINGLE: {desviacion_estandar:.2f}')


st.subheader('Monto personal')
input_text = st.text_input("Ingresa tu palabra ultra secreta y presiona Enter:")

# Verificar si la palabra ingresada es 'lokura'
if input_text.lower() == 'lokura':
    resultado = 1
    st.write(f"Tu monto a la fecha es de: {resultado}")
else:
    resultado = 0

