
# Time-Series

import functions as fun
import random
import gc

# Simulação para fator de potência unitário

def compile_fluxo(dss, dss_file, dia, hora, curvas_file, location, Percentagem_NPFV):
    dss.text(f"Compile [{dss_file}]")
    dss.text("edit Transformer.REG1A %loadloss=0.000001 xhl=0.0000001")
    dss.text(f"edit Vsource.SOURCE pu=1.0")
    dss.text("set mode=daily")
    dss.text("set stepsize=1h")
    dss.text("set number=1")
    dss.solution_write_hour(hora-1)

    # Definir semente dos sorteios
    random.seed(location)

    # Carregar e normalizar curvas
    df_curva_normalizada, dict_new_mults_loadshapes = fun.carregar_e_normalizar_loadshapes(curvas_file)

    # Criar os LoadShapes no OpenDSS
    fun.criar_loadshapes(dss, dict_new_mults_loadshapes)

    # Obter nomes dos loadshapes e cargas
    dailys = dss.loadshapes_all_names()
    dailys.remove('default')
    loads = dss.loads_all_names()

    # Sortear e atribuir curvas de carga
    curva_carga_dict = fun.sortear_curvas_para_cargas(dss, loads, dailys)

    # Identificar barras trifásicas e monofásicas com nome numérico e suas tensões
    tribuses_dict, monbuses_dict = fun.classificar_barras_por_fases(dss)

    loads_power_dict = fun.obter_potencias_das_cargas(dss)

    # Classificar cargas por tipo de barra
    cargas_tribuses, cargas_monbuses = fun.classificar_cargas_por_tipo_de_barra(loads_power_dict, tribuses_dict.keys())

    # Agrupar cargas que estão em barras trifásicas
    cargas_tribuses_agg = fun.agrupar_cargas_por_barra_trifasica(cargas_tribuses)

    # Lista de consumidores elegiveis para instalar SFVs
    all_loads_dict = {**cargas_tribuses_agg, **cargas_monbuses}

    # Obtecao das probabilidades ponderadas de cada carga segundo a sua potencia ativa instalada (kW)
    probabilidades = fun.calcular_probabilidades_ponderadas(all_loads_dict)

    # -------------------------------------------- DIMENSIONAMENTO DOS SFVs ------------------------------------------ #

    # Definir a potencia do inversor de acordo com a tabela do INMET
    # Lista de potências dos inversores disponíveis, ordenada do maior para o menor
    inversores_tri_disponiveis = [180, 150, 125, 100, 50, 30, 25, 20, 18, 15, 12, 10, 9, 8, 7, 6, 5]
    inversores_mon_disponiveis = [15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    # Cálculo da Pmpp ajustada com base em curva de carga
    KVA_all_inversores, Pmpp_all_consumidores = fun.dimensionar_inversores_consumidores(
    cargas_tribuses,
    cargas_monbuses,
    curva_carga_dict,
    df_curva_normalizada,
    inversores_tri_disponiveis,
    inversores_mon_disponiveis
)

    # Seleção dos consumidores
    consumidores_escolhidos, npfv_final = fun.atingir_npfv(KVA_all_inversores, probabilidades, Percentagem_NPFV, Pmpp_all_consumidores)

    # Instalação dos SFVs
    fun.instalar_sfv(dia, dss, consumidores_escolhidos, tribuses_dict, monbuses_dict)

    dss.solution_solve()

    # Resultados

    # Tensoes nodais
    buses_tensoes, v_max, v_min = fun.get_bus_node_voltages(dss)

    Tensoes_abc_pu = fun.Tensoes_abc_pu(dss)

    # Potências totais dos sistemas fotovoltaicos
    total_pv_p, total_pv_q, total_pv_p_dict, total_pv_q_dict = fun.get_total_pv_powers(dss)

    # Potências kW e KVA do circuito
    total_p_kw = (-1 * dss.circuit_total_power()[0])
    total_q_kvar = (-1 * dss.circuit_total_power()[1])

    # Perdas em kW circuito
    total_losses_p_kw = (dss.circuit_losses()[0] / 1000)  # kW

    # Registro das violacoes

    # Sobrecarga dos condutores
    OL_dict, num_SC, max_OL_CVV_PSO = fun.calcular_sobrecarga_linhas(dss)

    # Desequilibrio de tensao
    eliminar = ['150', '150r', '160r', '61s', '300_open', '25r', '9r', '94_open', '610']
    FD_dict, max_FD, dict_total_violations_DT, num_DT = fun.calcular_desequilibrio_tensao(dss, tribuses_dict, eliminar)

    # Violacoes dos limites de tensao
    OV_buses, UV_buses, num_OV, num_UV = fun.verificar_violacoes_tensao(dss, eliminar, lim_OV = 1.05,
                                                                        lim_UV = 0.95)

    del dss
    gc.collect()

    return num_OV, num_UV, num_SC, num_DT, buses_tensoes, Tensoes_abc_pu, v_max, v_min, total_pv_p, total_pv_q, \
        total_losses_p_kw, total_p_kw, total_q_kvar, total_pv_p_dict, total_pv_q_dict



