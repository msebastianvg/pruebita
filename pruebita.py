import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards

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

def formatear_miles(numero):
    return '{:,.0f}'.format(numero).replace(',', '.')


st.title('Reporte BETS - 2023')

st.subheader('Comienza el último periodo del año: 09 de Octubre hasta 31 de Diciembre.')









# Cargar los datos desde el archivo Excel
file_path = 'bets-2023-2.xlsx'
df = pd.read_excel(file_path, sheet_name='bets')
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d', errors='coerce')
df = df.sort_values(by='DATE')
df['Max_ID'] = df.groupby('DATE')['ID'].transform('max')
last_wl = df[df['ID'] == df['Max_ID']]

# Filtrar solo el último valor de "PERCENTAGE" para cada día
last_pozo_actual = last_wl.groupby('DATE')['PERCENTAGE'].last().reset_index()

# Aplicar la lógica de colores en función de 'WL'
last_wl['Color'] = 'lightgreen'
last_wl.loc[last_wl['WL'] == 0, 'Color'] = 'mistyrose'

# Combinar los DataFrames 'last_pozo_actual' y 'last_wl' para tener los colores
last_pozo_actual = last_pozo_actual.merge(last_wl[['DATE', 'Color']], on='DATE', how='left')
last_pozo_actual['DATE'] = last_pozo_actual['DATE'].dt.strftime('%Y-%m-%d')

fig = px.bar(
    last_pozo_actual,
    x='DATE',  # Especifica que DATE es el eje X
    y='PERCENTAGE',
    color='Color',
    color_discrete_map={'lightgreen': 'lightgreen', 'mistyrose': 'mistyrose'},
)

fig.update_yaxes(
    ticksuffix="%",
    range=[-2.5, 5]
)

fig.update_layout(
    xaxis_title='Fecha',
    yaxis_title='Porcentaje de ganancias (%)',
    xaxis=dict(
        type='category',
        categoryorder='category ascending'  # Ordenar por año, mes, día
    ),
    showlegend=False
)

st.plotly_chart(fig)








df = df.sort_values(by='ID', ascending=False)

# ultimo_percentage = df['PERCENTAGE'].iloc[0]

df['PERCENTAGE'] = df['PERCENTAGE'].astype(float)  # Asegúrate de que la columna sea de tipo float

# Eliminar filas con NaN
df_cleaned = df.dropna(subset=['PERCENTAGE'])

if not df_cleaned.empty:
    df_cleaned = df_cleaned.iloc[::-1] 
    ultimo_percentage = df_cleaned['PERCENTAGE'][df_cleaned['PERCENTAGE'] > -10].iloc[-1]
else:
    ultimo_percentage = None 



# penultimo_percentage = df['PERCENTAGE'].iloc[1]

for i in range(1, 11):
    valor = df['PERCENTAGE'].iloc[i]
    
    if not pd.isna(valor) and valor > 0:
        penultimo_percentage = valor
        break

ultima_perdida = df.loc[df['WL'] == 0, 'DATE'].max()

apuestas_ganadas_desde_ultima_perdida = df[(df['DATE'] > ultima_perdida) & (df['WL'] == 1)]['WL'].count()

fecha_maxima = df['DATE'].max()
apuestas_ganadas_ultimo_dia = df[(df['DATE'] == fecha_maxima) & (df['WL'] == 1)]
cantidad_apuestas_ganadas_ultimo_dia = len(apuestas_ganadas_ultimo_dia)



# Obtener el último registro con WL=0 ordenado por ID
ultimo_wl0 = df[df['WL'] == 0].sort_values(by='ID', ascending=False).iloc[0]

# Filtrar los registros con WL=1 después del último WL=0
apuestas_ganadas_desde_ultima_perdida = df[(df['WL'] == 1) & (df['ID'] > ultimo_wl0['ID'])]['ID'].count()

col1, col2 = st.columns(2)

label1 = "Porcentaje de ganancia actual"
value1 = f"{ultimo_percentage/100:.2%}"
delta1 = f"{(ultimo_percentage - penultimo_percentage)/100:.2%}" 

label2 = "Racha de victorias"
value2 = apuestas_ganadas_desde_ultima_perdida
delta2 = cantidad_apuestas_ganadas_ultimo_dia

# Crear el panel métrico
col1.metric(label=label1, value=value1, delta=delta1)
col2.metric(label=label2, value=value2, delta=delta2)



total_wins = (df['WL'] == 1).sum()
total_losses = (df[df['WL'] == 0]['WL'] == 0).sum()
media = total_wins/(total_losses+total_wins)
apuestas_ganadas = total_wins
apuestas_perdidas = total_losses

fig = px.bar(
    x=["Apuestas ganadas", "Apuestas perdidas"],
    y=[total_wins, total_losses],
    color=["Ganadas", "Perdidas"],
    labels={"x": "", "y": ""},
#    title="Apuestas Ganadas vs. Apuestas Perdidas",
    color_discrete_map={"Ganadas": "lightgreen", "Perdidas": "mistyrose"}
)
st.write(fig)

st.metric(label="Porcentaje de victorias", value=f"{total_wins/(total_losses+total_wins):.2%}" )



# Cargar los datos desde el archivo Excel
file_path = 'bets-2023-2.xlsx'
sheet_name = 'bets'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Agrupar por 'CATEGORY' y 'WL' y contar la cantidad de registros
grouped = df.groupby(['CATEGORY', 'WL']).size().unstack(fill_value=0)

# Reiniciar el índice para tener 'CATEGORY' como una columna
grouped = grouped.reset_index()

# Cambiar los nombres de las columnas
grouped = grouped.rename(columns={0: 'Apuestas perdidas', 1: 'Apuestas ganadas'})

# Crear un gráfico interactivo en Streamlit
fig = px.bar(
    grouped,
    x='CATEGORY',
    y=['Apuestas perdidas', 'Apuestas ganadas'],  # Corregir los nombres de las columnas
    barmode='group',
    # title='Cantidad de Registros por Categoría y WL',
    color_discrete_map={"Apuestas ganadas": "lightgreen", "Apuestas perdidas": "mistyrose"},  # Corregir la paleta de colores
)
fig.update_layout(
    xaxis_title='Categoría',
    yaxis_title='Cantidad de apuestas',
    showlegend=True,
)

fig.update_xaxes(categoryorder='total ascending')

st.plotly_chart(fig)





# Cargar los datos desde el archivo Excel
file_path = 'bets-2023-2.xlsx'
sheet_name = 'bets'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Agrupar por 'CATEGORY' y 'WL' y contar la cantidad de registros
grouped = df.groupby(['TEAM', 'WL']).size().unstack(fill_value=0)

# Reiniciar el índice para tener 'CATEGORY' como una columna
grouped = grouped.reset_index()

# Cambiar los nombres de las columnas
grouped = grouped.rename(columns={0: 'Apuestas perdidas', 1: 'Apuestas ganadas'})

# Crear un gráfico interactivo en Streamlit
fig = px.bar(
    grouped,
    x='TEAM',
    y=['Apuestas perdidas', 'Apuestas ganadas'],  # Corregir los nombres de las columnas
    barmode='group',
    # title='Cantidad de Registros por Categoría y WL',
    color_discrete_map={"Apuestas ganadas": "lightgreen", "Apuestas perdidas": "mistyrose"},  # Corregir la paleta de colores
)
fig.update_layout(
    xaxis_title='Equipo',
    yaxis_title='Cantidad de apuestas',
    showlegend=True,
)

fig.update_xaxes(categoryorder='total ascending')

st.plotly_chart(fig)





# Cargar los datos desde el archivo Excel (asegúrate de que tus datos tengan las columnas 'WL' y 'DATE')
file_path = 'bets-2023-2.xlsx'
df = pd.read_excel(file_path)

# Asegúrate de que 'DATE' sea de tipo datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Agregar una columna con el día de la semana (0: lunes, 1: martes, ..., 6: domingo)
df['Day_of_Week'] = df['DATE'].dt.dayofweek

# Calcular el porcentaje de apuestas ganadas para cada día de la semana
day_stats = df.groupby('Day_of_Week')['WL'].mean() * 100  # Multiplicar por 100 para obtener un porcentaje

# Crear un DataFrame para el gráfico
radar_data = pd.DataFrame({'Day_of_Week': day_stats.index, 'Win_Percentage': day_stats.values})

# Nombres de los días de la semana
day_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

# Asignar nombres de días a los valores del eje X
radar_data['Day_of_Week'] = radar_data['Day_of_Week'].map({i: day for i, day in enumerate(day_names)})

# Crear el gráfico de radar interactivo
fig = px.line_polar(radar_data, r='Win_Percentage', theta='Day_of_Week', line_close=True)

# Personalizar el color de relleno y el borde
fig.update_traces(
    line=dict(color='green'),
    fill='toself',  # Relleno del área bajo la curva
    fillcolor='rgba(144, 238, 144, 0.5)'  # Color de relleno lightgreen
)

st.plotly_chart(fig)





st.subheader('Monto personal')
input_text = st.text_input("Ingresa tu palabra ultra secreta y presiona Enter:")

df = pd.read_excel(file_path, sheet_name='resumen')
filtro_tipo_2 = df[df['TIPO'] == 4]
if not filtro_tipo_2.empty:
    valor_v = filtro_tipo_2.iloc[0]['V']
    valor_c = filtro_tipo_2.iloc[0]['C']
    valor_e = filtro_tipo_2.iloc[0]['E']
    valor_m = filtro_tipo_2.iloc[0]['M']
    valor_cc = filtro_tipo_2.iloc[0]['CC']

# Verificar si la palabra ingresada es 'lokura'
if input_text.lower() == 'ornn':
    valor_v_v = f"${valor_v:,.0f} CLP"
    st.write(f"Tu monto a la fecha es de: {valor_v_v.replace(',', '.')}")  
elif input_text.lower() == 'rufi':
    valor_e_e = f"${valor_e:,.0f} CLP"
    st.write(f"Tu monto a la fecha es de: {valor_e_e.replace(',', '.')}")  
elif input_text.lower() == 'morty':
    valor_c_c = f"${valor_c:,.0f} CLP"
    st.write(f"Tu monto a la fecha es de: {valor_c_c.replace(',', '.')}")  
elif input_text.lower() == 'duskelokura':
    valor_m_m = f"${valor_m:,.0f} CLP"
    st.write(f"Tu monto a la fecha es de: {valor_m_m.replace(',', '.')}")  
elif input_text.lower() == 'kombucha':
    valor_cc_cc = f"${valor_cc:,.0f} CLP"
    st.write(f"Tu monto a la fecha es de: {valor_cc_cc.replace(',', '.')}")  
else:
    resultado = 0



st.text(' ')
st.text(' ')

st.text('Resultados del periodo anterior (hasta el 08 de Octubre):')

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

fig = px.bar(
    x=["Apuestas ganadas", "Apuestas perdidas"],
    y=[total_wins, total_losses],
    color=["Ganadas", "Perdidas"],
    labels={"x": "", "y": ""},
#    title="Apuestas Ganadas vs. Apuestas Perdidas",
    color_discrete_map={"Ganadas": "lightgreen", "Perdidas": "mistyrose"},
)
st.write(fig)






df['DATE'] = pd.to_datetime(df['DATE'])

# Seleccionar el último registro de cada fecha
df = df.groupby('DATE').tail(1)

# Calcular el promedio global de 'PERCENTAGE'
average_single_global = df['PERCENTAGE'].mean()

# Asignar colores según 'WL'
color_map = {'0': 'red', 1: 'green'}

# Crear un gráfico interactivo utilizando Plotly Express
fig = px.bar(df, x='DATE', y='PERCENTAGE', color='WL', color_discrete_map=color_map, labels={'PERCENTAGE': 'GANANCIAS (%)'})
fig.update_traces(marker_line_width=0)  # Eliminar las líneas de borde
fig.update_layout(
    xaxis_title='FECHA',
    yaxis_title='GANANCIAS (%)',
    yaxis_tickformat='%',
    title='GANANCIAS (%) POR FECHA',
    showlegend=True,
)
fig.add_hline(y=average_single_global, line_dash='dash', line_color='blue', name='PROMEDIO DE GANANCIAS (%)')

# Mostrar el gráfico interactivo en Streamlit
st.plotly_chart(fig)










