# ativos_b3.py - Base de dados de ativos da B3 organizados por segmento

ACOES_B3 = {
    'Financeiro': [
        'BBAS3.SA', 'BBDC4.SA', 'ITUB4.SA', 'SANB11.SA', 'BBSE3.SA',
        'BPAC11.SA', 'ITSA4.SA', 'B3SA3.SA', 'CIEL3.SA', 'PSSA3.SA',
        'ABCB4.SA', 'BPAN4.SA', 'BRSR6.SA', 'PINE4.SA', 'BBDC3.SA'
    ],
    'Petróleo e Gás': [
        'PETR4.SA', 'PETR3.SA', 'PRIO3.SA', 'RRRP3.SA', 'RECV3.SA',
        'CSAN3.SA', 'UGPA3.SA', 'ENGI11.SA'
    ],
    'Mineração e Siderurgia': [
        'VALE3.SA', 'CSNA3.SA', 'USIM5.SA', 'GOAU4.SA', 'GGBR4.SA',
        'CMIN3.SA', 'SUZB3.SA'
    ],
    'Energia Elétrica': [
        'EGIE3.SA', 'CPLE6.SA', 'CPFE3.SA', 'ELET3.SA', 'ELET6.SA',
        'TAEE11.SA', 'NEOE3.SA', 'ENEV3.SA', 'CMIG4.SA', 'TRPL4.SA',
        'ENGI11.SA', 'AURE3.SA', 'AESB3.SA', 'COCE5.SA'
    ],
    'Telecomunicações': [
        'VIVT3.SA', 'TIMS3.SA', 'OIBR3.SA', 'OIBR4.SA', 'TIET11.SA'
    ],
    'Consumo': [
        'ABEV3.SA', 'JBSS3.SA', 'BEEF3.SA', 'MRFG3.SA', 'SMTO3.SA',
        'CRFB3.SA', 'PCAR3.SA', 'NTCO3.SA', 'SLCE3.SA', 'CAML3.SA',
        'RAIZ4.SA', 'SOJA3.SA', 'AGRO3.SA'
    ],
    'Varejo': [
        'MGLU3.SA', 'LREN3.SA', 'ARZZ3.SA', 'PETZ3.SA', 'VIVA3.SA',
        'SOMA3.SA', 'CEAB3.SA', 'GUAR3.SA', 'ALPA4.SA', 'BHIA3.SA',
        'LCAM3.SA', 'GRND3.SA'
    ],
    'Construção Civil': [
        'CYRE3.SA', 'MRVE3.SA', 'EZTC3.SA', 'TEND3.SA', 'DIRR3.SA',
        'LAVV3.SA', 'PLPL3.SA', 'JHSF3.SA', 'EVEN3.SA', 'HBOR3.SA'
    ],
    'Saúde': [
        'RADL3.SA', 'HAPV3.SA', 'GNDI3.SA', 'FLRY3.SA', 'PNVL3.SA',
        'QUAL3.SA', 'ODPV3.SA', 'HYPE3.SA', 'MATD3.SA', 'DASA3.SA'
    ],
    'Logística e Transporte': [
        'RAIL3.SA', 'CCRO3.SA', 'AZUL4.SA', 'GOLL4.SA', 'EMBR3.SA',
        'ECOR3.SA', 'TGMA3.SA', 'LOGN3.SA', 'STBP3.SA'
    ],
    'Tecnologia': [
        'TOTS3.SA', 'LWSA3.SA', 'POSI3.SA', 'IFCM3.SA', 'CASH3.SA',
        'SQIA3.SA', 'TECN3.SA', 'SEQL3.SA'
    ],
    'Indústria': [
        'WEGE3.SA', 'RENT3.SA', 'RAIZ4.SA', 'KLBN11.SA', 'SUZB3.SA',
        'EMBR3.SA', 'BRFS3.SA', 'POMO4.SA', 'FESA4.SA', 'KEPL3.SA',
        'LEVE3.SA', 'TUPY3.SA', 'ROMI3.SA', 'MYPK3.SA'
    ],
    'Papel e Celulose': [
        'SUZB3.SA', 'KLBN11.SA', 'SANB11.SA'
    ],
    'Educação': [
        'COGN3.SA', 'YDUQ3.SA', 'ANIM3.SA', 'SEER3.SA', 'VITB3.SA'
    ],
    'Agronegócio': [
        'SLCE3.SA', 'SOJA3.SA', 'AGRO3.SA', 'BEEF3.SA', 'JBSS3.SA'
    ]
}

FIIS = {
    'Shoppings': [
        'HGBS11.SA', 'XPML11.SA', 'VISC11.SA', 'MALL11.SA', 'HSML11.SA',
        'SHPH11.SA', 'SHDP11.SA'
    ],
    'Logística': [
        'HGLG11.SA', 'BTLG11.SA', 'LVBI11.SA', 'VILG11.SA', 'RBLG11.SA',
        'XPLG11.SA', 'BRCO11.SA'
    ],
    'Lajes Corporativas': [
        'HGRE11.SA', 'KNRI11.SA', 'PVBI11.SA', 'RBRR11.SA', 'RECT11.SA',
        'RZTR11.SA', 'GTWR11.SA', 'CVBI11.SA'
    ],
    'Títulos e Valores Mobiliários': [
        'MXRF11.SA', 'KNCR11.SA', 'BTCI11.SA', 'HGCR11.SA', 'VCRI11.SA',
        'VGIR11.SA', 'RZAK11.SA'
    ],
    'Híbridos': [
        'HGRU11.SA', 'XPPR11.SA', 'BRCO11.SA', 'VIUR11.SA', 'JSRE11.SA'
    ],
    'Residencial': [
        'HGPO11.SA', 'RBVA11.SA', 'VRTA11.SA', 'KNIP11.SA', 'VILG11.SA'
    ],
    'Hotéis': [
        'HTMX11.SA', 'XPHT11.SA'
    ],
    'Agronegócio': [
        'GGRC11.SA', 'AGCX11.SA', 'RZAG11.SA'
    ]
}

ETFS = {
    'Índices Brasileiros': [
        'BOVA11.SA', 'SMAL11.SA', 'IVVB11.SA', 'FIND11.SA', 'MATB11.SA',
        'PIBB11.SA', 'DIVO11.SA', 'GOVE11.SA'
    ],
    'Índices Internacionais': [
        'IVVB11.SA', 'WRLD11.SA', 'NASD11.SA', 'SPXI11.SA', 'ESGB11.SA'
    ],
    'Setoriais': [
        'ISUS11.SA', 'FIND11.SA', 'UTIL11.SA', 'MATB11.SA', 'HASH11.SA',
        'NFTS11.SA'
    ],
    'Renda Fixa': [
        'IMAB11.SA', 'IMA211.SA', 'FIXA11.SA', 'B5P211.SA', 'IRFM11.SA'
    ],
    'Smart Beta': [
        'DIVO11.SA', 'GOVE11.SA', 'BOVV11.SA'
    ],
    'Temáticos': [
        'HASH11.SA', 'NFTS11.SA', 'ESGB11.SA', 'ISUS11.SA', 'COIN11.SA'
    ]
}

BDRS = {
    'Tecnologia': [
        'AAPL34.SA', 'MSFT34.SA', 'GOOGL34.SA', 'AMZO34.SA', 'META34.SA',
        'NVDC34.SA', 'TSLA34.SA', 'NFLX34.SA'
    ],
    'Financeiro': [
        'BOAC34.SA', 'C1IT34.SA', 'JPMC34.SA', 'GSGI34.SA', 'VISA34.SA',
        'MSTR34.SA'
    ],
    'Consumo': [
        'COCA34.SA', 'PEPB34.SA', 'WALM34.SA', 'NIKE34.SA', 'SBUB34.SA',
        'M1CD34.SA'
    ],
    'Saúde': [
        'PFIZ34.SA', 'JHNS34.SA', 'ABBV34.SA', 'MRCK34.SA'
    ],
    'Diversos': [
        'DISB34.SA', 'BERK34.SA', 'XPBR31.SA'
    ]
}

def get_all_tickers():
    """Retorna todos os tickers disponíveis"""
    all_tickers = []
    
    for categoria in ACOES_B3.values():
        all_tickers.extend(categoria)
    
    for categoria in FIIS.values():
        all_tickers.extend(categoria)
    
    for categoria in ETFS.values():
        all_tickers.extend(categoria)
    
    for categoria in BDRS.values():
        all_tickers.extend(categoria)
    
    return list(set(all_tickers))

def get_tickers_by_type(tipo):
    """Retorna tickers por tipo (acoes, fiis, etfs, bdrs)"""
    tipo_map = {
        'acoes': ACOES_B3,
        'fiis': FIIS,
        'etfs': ETFS,
        'bdrs': BDRS
    }
    
    if tipo.lower() in tipo_map:
        all_tickers = []
        for categoria in tipo_map[tipo.lower()].values():
            all_tickers.extend(categoria)
        return list(set(all_tickers))
    
    return []
