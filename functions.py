import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # pip install scikit-learn
import random
import math

def carregar_e_normalizar_loadshapes(curvas_file):
    """
    Carrega um arquivo CSV contendo curvas de demanda horária, normaliza os dados
    para o intervalo [0, 1], e retorna um dicionário com os nomes das curvas como
    chaves e listas de valores normalizados como valores.

    Parâmetros:
    -----------
    curvas_file : str
        Caminho completo para o arquivo CSV com as curvas de demanda.

    Retorna:
    --------
    dict_loadshapes : dict
        Dicionário no formato {nome_curva: [valores_normalizados]}
    """
    # Carregar o arquivo CSV
    df = pd.read_csv(curvas_file)

    # Converter nomes das colunas para minúsculas (consistência)
    df.columns = df.columns.str.lower()

    # Normalizar os dados para o intervalo [0, 1]
    scaler = MinMaxScaler()
    df_curva_normalizada = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

    # Criar dicionário com os valores normalizados por curva
    dict_loadshapes = {
        coluna: df_curva_normalizada[coluna].tolist()
        for coluna in df_curva_normalizada.columns
    }

    return df_curva_normalizada, dict_loadshapes

def criar_loadshapes(dss, dict_loadshapes):
    """
    Cria objetos LoadShape no OpenDSS com base em um dicionário de curvas.

    Parâmetros:
    -----------
    dss : objeto OpenDSS
        Interface com o motor OpenDSS.
    dict_loadshapes : dict
        Dicionário {nome_loadshape: lista de valores mult}.
    """
    for nome, mult in dict_loadshapes.items():
        dss.text(f"New Loadshape.{nome} npts=96 interval=0.25 mult={mult}")

def sortear_curvas_para_cargas(dss, loads, dailys):
    """
    Atribui aleatoriamente curvas de carga (loadshapes) a cada carga do sistema.

    Parâmetros:
    -----------
    dss : objeto OpenDSS
    loads : list
        Lista de nomes das cargas.
    dailys : list
        Lista de nomes de loadshapes disponíveis (exceto 'default').

    Retorna:
    --------
    curva_carga_dict : dict
        Dicionário {nome_da_carga: loadshape_atribuído}
    """
    curva_carga_dict = {}

    dss.loads_first()
    for load in loads:
        daily = random.choice(dailys)
        dss.loads_write_daily(daily)
        curva_carga_dict[load] = daily
        dss.loads_next()

    return curva_carga_dict

def classificar_barras_por_fases(dss):
    """
    Classifica as barras do circuito em trifásicas e monofásicas, excluindo a 'sourcebus',
    e retorna dois dicionários contendo apenas as barras com nomes numéricos.

    Parâmetros:
    -----------
    dss : objeto OpenDSS
        Interface com o motor OpenDSS.

    Retorna:
    --------
    tribuses_dict : dict
        Dicionário com barras trifásicas e suas tensões base (somente nomes numéricos).

    monbuses_dict : dict
        Dicionário com barras monofásicas e suas tensões base (somente nomes numéricos).
    """
    tribus_voltages_dict = {}
    monbus_voltages_dict = {}

    for bus in dss.circuit_all_bus_names():
        if bus.lower() == 'sourcebus':
            continue

        dss.circuit_set_active_bus(bus)
        kv_bus = dss.bus_kv_base() # LN Voltage
        num_phases = len(dss.bus_nodes())

        if kv_bus > 1.0:
            if num_phases == 3:
                tribus_voltages_dict[bus] = kv_bus
            elif num_phases == 1:
                monbus_voltages_dict[bus] = kv_bus

    # Filtrar apenas as barras com nomes numéricos (ex: '76', '112')
    tribuses_dict = {key: value for key, value in tribus_voltages_dict.items() if key.isdigit()}
    monbuses_dict = {key: value for key, value in monbus_voltages_dict.items() if key.isdigit()}

    return tribuses_dict, monbuses_dict


def obter_potencias_das_cargas(dss):
    """
    Lê a potência ativa (kW) de todas as cargas do circuito no OpenDSS.

    Parâmetros:
    -----------
    dss : objeto OpenDSS
        Interface com o motor OpenDSS.

    Retorna:
    --------
    loads_power_dict : dict
        Dicionário no formato {nome_da_carga: potencia_kw}
    """
    loads_power_dict = {}
    loads = dss.loads_all_names()

    dss.loads_first()
    for load in loads:
        loads_power_dict[load] = dss.loads_read_kw()
        dss.loads_next()

    return loads_power_dict

def classificar_cargas_por_tipo_de_barra(loads_power_dict, tribus_keys):
    """
    Classifica as cargas de acordo com o tipo de barra (trifásica ou monofásica).

    Parâmetros:
    -----------
    loads_power_dict : dict
        Dicionário {nome_carga: potência_kW}
    tribus_keys : iterável
        Lista ou conjunto com os nomes das barras trifásicas

    Retorna:
    --------
    cargas_tri : dict
        Cargas conectadas a barras trifásicas
    cargas_mon : dict
        Cargas conectadas a barras monofásicas
    """
    cargas_tri = {}
    cargas_mon = {}

    for load_name, power in loads_power_dict.items():
        bus_number = ''.join([char for char in load_name if char.isdigit()])

        if bus_number in tribus_keys:
            cargas_tri[load_name] = power
        else:
            cargas_mon[load_name] = power

    return cargas_tri, cargas_mon

def agrupar_cargas_por_barra_trifasica(cargas_tribuses):
    """
    Agrupa cargas que estão conectadas à mesma barra, somando suas potências
    e registrando todas as fases presentes (a, b, c).

    Exemplo:
        Se existirem as cargas:
            'load76a': 50.0
            'load76b': 60.0
            'load76c': 70.0
        A barra 76 receberá:
            's76abc': 180.0

    Parâmetros:
    -----------
    cargas_tribuses : dict
        Dicionário {nome_carga: potência_kW} com cargas conectadas a barras trifásicas.

    Retorna:
    --------
    cargas_agregadas : dict
        Dicionário no formato {'s<numero_barra><fases_agregadas>': soma_potencias}
        Ex: {'s76abc': 180.0}
    """
    cargas_por_barra = {}

    for nome_carga, potencia in cargas_tribuses.items():
        # Extrai o número da barra (ex: 76 de "load76a")
        numero_barra = ''.join(filter(str.isdigit, nome_carga))
        # Extrai a(s) letra(s) da fase (ex: "a" de "load76a")
        fase = ''.join(filter(str.isalpha, nome_carga))

        if numero_barra not in cargas_por_barra:
            cargas_por_barra[numero_barra] = {
                'fases': set(),
                'pot_total': 0.0
            }

        cargas_por_barra[numero_barra]['fases'].add(fase)
        cargas_por_barra[numero_barra]['pot_total'] += potencia

    # Monta o dicionário final com chave no formato 's<num_barra><fases>'
    cargas_agregadas = {
        f"s{barra}{''.join(sorted(info['fases']))}": info['pot_total']
        for barra, info in cargas_por_barra.items()
    }

    return cargas_agregadas


def calcular_probabilidades_ponderadas(cargas_dict):
    """
    Calcula a probabilidade ponderada de cada carga com base na sua potência ativa.

    Parâmetros:
    -----------
    cargas_dict : dict
        Dicionário {nome_carga: potência_kW}

    Retorna:
    --------
    probabilidades : dict
        Dicionário {nome_carga: probabilidade}
    """
    soma_total = sum(cargas_dict.values())
    if soma_total == 0:
        raise ValueError("Soma das potências é zero. Não é possível calcular probabilidades.")

    return {chave: valor / soma_total for chave, valor in cargas_dict.items()}

import numpy as np
import math

def calcular_potencia_pmpp(demanda_maxima_kw, curva_normalizada, td=0.8, hsp=5.4, potencia_modulo_w=665):
    """
    Calcula a potência Pmpp (kW) de um sistema fotovoltaico com base no consumo diário
    estimado e ajusta para ser múltiplo da potência de um módulo solar.

    Parâmetros:
    -----------
    demanda_maxima_kw : float
        Demanda máxima da carga em kW.

    curva_normalizada : list or array
        Curva de demanda normalizada (96 pontos representando 24h com intervalo de 15 minutos).

    td : float, opcional
        Taxa de desempenho do sistema (default = 0.8).

    hsp : float, opcional
        Horas de sol pleno por dia (default = 5.4).

    potencia_modulo_w : float, opcional
        Potência nominal de um painel solar em watts (default = 665 W).

    Retorna:
    --------
    pmpp_ajustada_kw : float
        Potência Pmpp ajustada em kW (múltiplo da potência do módulo).

    numero_modulos : int
        Número de módulos necessários para atingir a Pmpp ajustada.
    """
    # Energia total consumida no dia (kWh) via integração numérica da curva
    energia_total_kwh = demanda_maxima_kw * np.trapz(curva_normalizada, dx=0.25)

    # Pmpp inicial sem ajuste
    pmpp_inicial_kw = energia_total_kwh / (td * hsp)

    # Ajustar Pmpp para ser múltiplo da potência de um painel
    numero_modulos = math.ceil((pmpp_inicial_kw * 1000) / potencia_modulo_w)
    pmpp_ajustada_kw = numero_modulos * potencia_modulo_w / 1000

    return pmpp_ajustada_kw, numero_modulos

import math

def dimensionar_inversores_consumidores(
    cargas_tribuses,
    cargas_monbuses,
    curva_carga_dict,
    df_curva_normalizada,
    inversores_tri_disponiveis,
    inversores_mon_disponiveis
):
    """
    Para cada consumidor, calcula a potência Pmpp desejada com base na curva de demanda e demanda máxima.
    Em seguida, define o conjunto de inversores necessários, respeitando as potências disponíveis.

    Retorna:
    - KVA_inv_consumidores_tri: dict {barra_trifásica: [lista de inversores]}
    - KVA_inv_consumidores_mon: dict {carga_monofásica: [lista de inversores]}
    - Pmpp_all_consumidores: dict {consumidor ou barra: Pmpp (kW)}
    """

    # --- TRIFÁSICOS ---
    pmpp_tri = {}
    for load in cargas_tribuses:
        mult_daily = curva_carga_dict[load]
        curva = df_curva_normalizada[mult_daily]
        Pmpp_ajustada, _ = calcular_potencia_pmpp(cargas_tribuses[load], curva)
        pmpp_tri[load] = math.ceil(Pmpp_ajustada)

    # Agrupar cargas trifásicas por barra
    pmpp_tri_agg = agrupar_cargas_por_barra_trifasica(pmpp_tri)

    # --- MONOFÁSICOS ---
    pmpp_mon = {}
    for load in cargas_monbuses:
        mult_daily = curva_carga_dict[load]
        curva = df_curva_normalizada[mult_daily][0:96:4]
        Pmpp_ajustada, _ = calcular_potencia_pmpp(cargas_monbuses[load], curva)
        pmpp_mon[load] = math.ceil(Pmpp_ajustada)

    # --- DEFINIÇÃO DOS INVERSORES ---
    def selecionar_inversores(pmpp_desejado, lista_inversores):
        inversores_usados = []
        kva_total = 0

        for kva in lista_inversores:
            while kva_total + kva <= pmpp_desejado:
                inversores_usados.append(kva)
                kva_total += kva

        restante = pmpp_desejado - kva_total
        for kva in reversed(lista_inversores):
            if kva <= restante:
                inversores_usados.append(kva)
                kva_total += kva
                break

        return inversores_usados

    # Aplicar para trifásicos
    KVA_inv_consumidores_tri = {
        consumidor: selecionar_inversores(pmpp, inversores_tri_disponiveis)
        for consumidor, pmpp in pmpp_tri_agg.items()
    }

    # Aplicar para monofásicos
    KVA_inv_consumidores_mon = {
        consumidor: selecionar_inversores(pmpp, inversores_mon_disponiveis)
        for consumidor, pmpp in pmpp_mon.items()
    }

    # Junta os dicionários de inversores trifásicos e monofásicos
    KVA_all_inversores = {**KVA_inv_consumidores_tri, **KVA_inv_consumidores_mon}

    # Juntar todos os Pmpp
    Pmpp_all_consumidores = {**pmpp_mon, **pmpp_tri_agg}

    return KVA_all_inversores, Pmpp_all_consumidores


# Seleciona aleatoriamente um consumidor com base nas probabilidades
def selecionar_aleatoriamente_com_probabilidades(probabilidades):
    return random.choices(list(probabilidades.keys()), weights=probabilidades.values(), k=1)[0]

# Função para atingir o NPFV desejado
def atingir_npfv(KVA_all_inversores, probabilidades, Percentagem_NPFV, Pmpp_all_consumidores):
    npfv_atual = 0
    consumidores_escolhidos = {}
    # Máxima Potência Instalada dos PVSystems
    Max_Pot_all_pvsytems = sum(Pmpp_all_consumidores.values())  # kW
    # Definição do Nível de Penetração Fovotoltaica (NPFV)
    npfv_desejado = (Percentagem_NPFV / 100) * Max_Pot_all_pvsytems  # Kw

    # Caso Percentagem_NPFV seja até 100%
    if Percentagem_NPFV < 100:
        consumidores_selecionados = set()

        while npfv_atual < npfv_desejado:
            consumidor = selecionar_aleatoriamente_com_probabilidades(probabilidades)

            if consumidor in consumidores_selecionados:
                continue

            consumidores_selecionados.add(consumidor)
            potencia = KVA_all_inversores[consumidor]
            potencia_total = sum(potencia) if isinstance(potencia, list) else float(potencia)

            npfv_atual += potencia_total
            consumidores_escolhidos[consumidor] = potencia_total

    # Caso Percentagem_NPFV seja maior que 100%
    else:
        consumidores_escolhidos = Pmpp_all_consumidores.copy()

    return consumidores_escolhidos, npfv_atual

# Para SFV_3ph e SFV_1ph  preciso de Nome_carga - Pmpp - Tensao - Nome_da_barra
def define_1ph_pvsystem(dia, dss, bus, kv, kva, pmpp):
    dss.text("New XYCurve.MyPvsT npts=4  xarray=[0  25  75  100]  yarray=[1 1 1 1]")
    dss.text("New XYCurve.MyEff npts=4  xarray=[.1  .2  .4  1.0]  yarray=[1 1 1 1]")

    if dia == 'SOL':
        # Dia Ensolarado
        dss.text(
            "New Loadshape.MyIrrad npts=24 interval=1 "
            "mult=[0.0,0.0,0.0,0.0,0.0,0.0,0.05,0.37,0.69,0.84,0.95,1.0,0.97,0.88,0.75,0.58,0.29,0.03,0.0,0.0,0.0,0.0,0.0,0.0]")
        dss.text(
            "New Tshape.MyTemp npts=24 interval=1 temp=[26.6 26.6 26.4 26.2 25.5 25.1 24.9 25 26.1 29.2 30.6 31.8 31.6 31.8 31.9 31.7 31.8 30.3 28.7 27.5 27.2 27 26.7 26.6]")

    elif dia == 'CHUVA':
        # Dia Chuvoso
        dss.text(
            "New Loadshape.MyIrrad npts=24 interval=1 mult=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0728, 0.20720000000000002, 0.28, 0.56, 0.15120000000000003, 0.27440000000000003, 0.35840000000000005, 0.35840000000000005, 0.19040000000000004, 0.044800000000000006, 0.033600000000000005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]")
        dss.text(
            "New Tshape.MyTemp npts=24 interval=1 temp=[25.6 25.3 25.2 23.2 23.7 23.7 23.8 23.7 23.4 24.2 24.8 25.6 26.4 24.6 25 24.9 27.5 25.7 25.1 24.4 24.9 24.1 24.9 25.3]")

    dss.text(
        f"New PVSystem.PV_{bus} phases=1 conn=wye bus1={bus} kV={kv} kVA={kva} Pmpp={pmpp} pf=1 effcurve=Myeff "
        f"P-TCurve=MyPvsT Daily=MyIrrad TDaily=MyTemp %cutin=0.00001 %cutout=0.00001 wattpriority=yes")

def define_3ph_pvsystem(dia, dss, bus, kv, kva, pmpp):
    dss.text("New XYCurve.MyPvsT npts=4  xarray=[0  25  75  100]  yarray=[1 1 1 1]")
    dss.text("New XYCurve.MyEff npts=4  xarray=[.1  .2  .4  1.0]  yarray=[1 1 1 1]")

    if dia == 'SOL':
        # Dia Ensolarado
        dss.text(
            "New Loadshape.MyIrrad npts=24 interval=1 "
            "mult=[0.0,0.0,0.0,0.0,0.0,0.0,0.05,0.37,0.69,0.84,0.95,1.0,0.97,0.88,0.75,0.58,0.29,0.03,0.0,0.0,0.0,0.0,0.0,0.0]")
        dss.text(
            "New Tshape.MyTemp npts=24 interval=1 temp=[26.6 26.6 26.4 26.2 25.5 25.1 24.9 25 26.1 29.2 30.6 31.8 31.6 31.8 31.9 31.7 31.8 30.3 28.7 27.5 27.2 27 26.7 26.6]")

    elif dia == 'CHUVA':
        # Dia Chuvoso
        dss.text(
            "New Loadshape.MyIrrad npts=24 interval=1 mult=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0728, 0.20720000000000002, 0.28, 0.56, 0.15120000000000003, 0.27440000000000003, 0.35840000000000005, 0.35840000000000005, 0.19040000000000004, 0.044800000000000006, 0.033600000000000005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]")
        dss.text(
            "New Tshape.MyTemp npts=24 interval=1 temp=[25.6 25.3 25.2 23.2 23.7 23.7 23.8 23.7 23.4 24.2 24.8 25.6 26.4 24.6 25 24.9 27.5 25.7 25.1 24.4 24.9 24.1 24.9 25.3]")

    dss.text(
        f"New PVSystem.PV_{bus} phases=3 conn=wye bus1={bus} kV={kv} kVA={kva} Pmpp={pmpp} pf=1 effcurve=Myeff "
        f"P-TCurve=MyPvsT Daily=MyIrrad TDaily=MyTemp %cutin=0.00001 %cutout=0.00001 wattpriority=yes")


def instalar_sfv(dia, dss, consumidores_escolhidos, tribuses_dict, monbuses_dict):
    """
    Instala os sistemas fotovoltaicos (SFVs) nos consumidores escolhidos.

    Parâmetros:
        dia (int): Dia da simulação.
        dss: Objeto do OpenDSS.
        consumidores_escolhidos (dict): Dicionário com nome do consumidor e valor de Pmpp.
        tribuses_dict (dict): Dicionário com barramentos trifásicos e suas tensões nominais.
        monbuses_dict (dict): Dicionário com barramentos monofásicos e suas tensões nominais.
    """
    for consumidor, pmpp in consumidores_escolhidos.items():
        bus_number = ''.join(filter(str.isdigit, consumidor))

        if bus_number in tribuses_dict:
            kv = tribuses_dict[bus_number]
            define_3ph_pvsystem(dia, dss, bus_number, kv, pmpp, pmpp)
        else:
            # Identificar a fase monofásica ('a', 'b' ou 'c') no final do nome do consumidor
            node = consumidor[-1].lower()
            phase_dict = {'a': '.1', 'b': '.2', 'c': '.3'}
            bus_with_node = f"{bus_number}{phase_dict.get(node, '.1')}"  # padrão: fase A

            kv = monbuses_dict[bus_number]
            define_1ph_pvsystem(dia, dss, bus_with_node, kv, pmpp, pmpp)



def get_bus_node_voltages(dss):
    """
    Retorna um dicionário com as tensões por fase de cada barra (barras numéricas, exceto 610 e 150).
    """
    # Filtra apenas barras com nome numérico
    all_buses = [bus for bus in dss.circuit_all_bus_names() if bus.isdigit()]
    for excluida in ['610', '150']:
        if excluida in all_buses:
            all_buses.remove(excluida)

    # Obtem os nomes dos nós e suas tensões
    nodes = dss.circuit_all_node_names()
    voltages = dss.circuit_all_bus_vmag_pu()

    # Mapeia os nós às suas tensões
    node_voltages = {
        node: voltage
        for node, voltage in zip(nodes, voltages)
        if voltage != 0.0
    }

    # Associa as tensões por fase a cada barra
    buses_tensoes = {}
    for bus_name in all_buses:
        buses_tensoes[bus_name] = [
            node_voltages.get(f"{bus_name}.1"),
            node_voltages.get(f"{bus_name}.2"),
            node_voltages.get(f"{bus_name}.3"),
        ]

    v_max = max(node_voltages.values())
    v_min = min(node_voltages.values())

    return buses_tensoes, v_max, v_min

def Tensoes_abc_pu(dss):
    Tensoes_abc_pu = []
    for bus in dss.circuit_all_bus_names():
        dss.circuit_set_active_bus(bus)
        name_bus = dss.bus_name()
        pu_bus = dss.bus_pu_vmag_angle()

        # Inicializando pu_list com zeros para os índices ausentes
        pu_list = [
            pu_bus[0] if len(pu_bus) > 0 else 0,
            pu_bus[2] if len(pu_bus) > 2 else 0,
            pu_bus[4] if len(pu_bus) > 4 else 0
        ]
        Tensoes_abc_pu.append(pu_list)

    return Tensoes_abc_pu

def get_total_pv_powers(dss):
    """
    Retorna as potências totais ativa (kW) e reativa (kVAr) de todos os PVsystems,
    além de dicionários com as potências por PV.
    """
    total_pv_p_list = []
    total_pv_q_list = []

    total_pv_p_dict = {}
    total_pv_q_dict = {}

    dss.pvsystems_first()
    for _ in range(dss.pvsystems_count()):
        name = dss.pvsystems_read_name()
        dss.circuit_set_active_element(f"PVsystem.{name}")
        powers = dss.cktelement_powers()

        # Potência ativa e reativa total (negativo porque injeção)
        p_total = -sum(powers[0:6:2])  # kW
        q_total = -sum(powers[1:6:2])  # kVAr

        total_pv_p_list.append(p_total)
        total_pv_q_list.append(q_total)

        total_pv_p_dict[name] = p_total
        total_pv_q_dict[name] = q_total

        dss.pvsystems_next()

    return sum(total_pv_p_list), sum(total_pv_q_list), total_pv_p_dict, total_pv_q_dict

def calcular_sobrecarga_linhas(dss):
    num_SC = 0
    OL_dict = {}

    dss.lines_first()
    for line in range(dss.lines_count()):
        dss.circuit_set_active_element(f"line.{dss.lines_read_name()}")
        current = dss.cktelement_currents_mag_ang()
        rating_current = dss.lines_read_norm_amps()

        if rating_current == 0:
            OL_dict[line] = 0
        else:
            OL_dict[line] = (max(current[0:12:2]) / rating_current) * 100

        if max(current[0:12:2]) > rating_current:
            num_SC += 1

        dss.lines_next()

    max_OL_CVV_PSO = max(OL_dict.values())
    return OL_dict, num_SC, max_OL_CVV_PSO


def calcular_desequilibrio_tensao(dss, tribuses_dict, eliminar):
    FD_dict = {}

    for bus in tribuses_dict.keys():
        dss.circuit_set_active_bus(bus)
        seq_voltages = dss.bus_seq_voltages()
        if abs(seq_voltages[1]) != 0:
            FD_dict[bus] = abs(seq_voltages[2]) / abs(seq_voltages[1]) * 100
        else:
            FD_dict[bus] = 0

    max_FD = max(FD_dict.values())

    buses_DT = {k: v for k, v in FD_dict.items() if k not in eliminar}
    dict_total_violations_DT = {k: v for k, v in buses_DT.items() if v > 2}
    num_DT = len(dict_total_violations_DT)

    return FD_dict, max_FD, dict_total_violations_DT, num_DT

def verificar_violacoes_tensao(dss, eliminar=None, lim_OV=1.05, lim_UV=0.95):
    """
    Verifica violações de sobretensão (OV) e subtensão (UV) por barra com base nas tensões nodais.

    Parâmetros:
        dss: objeto DSS (OpenDSS)
        eliminar (list): lista de barras a excluir da análise
        lim_OV (float): limite superior de tensão (pu) para considerar sobretensão
        lim_UV (float): limite inferior de tensão (pu) para considerar subtensão

    Retorna:
        OV_buses (list): lista de barras com sobretensão
        UV_buses (list): lista de barras com subtensão
        num_OV (int): número de barras com sobretensão
        num_UV (int): número de barras com subtensão
        OV_buses_voltages (dict): tensões por nodo com sobretensão
        UV_buses_voltages (dict): tensões por nodo com subtensão
    """

    if eliminar is None:
        eliminar = ['150', '150r', '160r', '61s', '300_open', '25r', '9r', '94_open', '610']

    nodes_with_OV = {}
    nodes_with_UV = {}

    OV_buses_voltages = {}
    UV_buses_voltages = {}

    for node, voltage in zip(dss.circuit_all_node_names(), dss.circuit_all_bus_vmag_pu()):
        nodes_with_OV[node] = voltage > lim_OV
        nodes_with_UV[node] = 0 < voltage < lim_UV

        if voltage > lim_OV:
            OV_buses_voltages[node] = voltage
        elif 0 < voltage < lim_UV:
            UV_buses_voltages[node] = voltage

    # Agrupamento por barras
    buses_with_OV = {}
    buses_with_UV = {}

    for node, ov in nodes_with_OV.items():
        bar, _ = node.split('.')
        buses_with_OV[bar] = buses_with_OV.get(bar, False) or ov

    for node, uv in nodes_with_UV.items():
        bar, _ = node.split('.')
        buses_with_UV[bar] = buses_with_UV.get(bar, False) or uv

    # Filtrar barras inválidas
    buses_OV = {k: v for k, v in buses_with_OV.items() if k not in eliminar}
    buses_UV = {k: v for k, v in buses_with_UV.items() if k not in eliminar}

    # Contagem de violações
    num_OV = sum(v for v in buses_OV.values())
    num_UV = sum(v for v in buses_UV.values())

    # Listas de barras com violações
    OV_buses = [k for k, v in buses_OV.items() if v]
    UV_buses = [k for k, v in buses_UV.items() if v]

    return OV_buses, UV_buses, num_OV, num_UV
