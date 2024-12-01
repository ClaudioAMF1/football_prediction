# Sistema de Previsão do Brasileirão

## 📋 Sobre o Projeto
Sistema de análise e previsão de resultados do Campeonato Brasileiro Série A utilizando Machine Learning. Desenvolvido como projeto final da disciplina de Machine Learning.

## ⚙️ Funcionalidades
- Previsão de resultados de partidas
- Análise estatística do campeonato
- Visualização de desempenho dos times
- Dashboard interativo
- Atualização automática de dados

## 🛠️ Tecnologias
- Python 3.10+
- Scikit-learn para Machine Learning
- Streamlit para interface web
- Plotly para visualizações
- API Football-data.org para dados em tempo real

## 📦 Estrutura do Projeto
```
football_prediction/
├── .cadence/              # Configurações do cadence
├── config/
│   └── config.yaml        # Arquivo de configuração
├── data/
│   ├── brasileirao_matches.csv  # Dados das partidas
│   └── classificacao.csv        # Tabela de classificação
├── logs/
│   └── data_collection.log      # Logs de coleta de dados
├── models/
│   └── brasileirao_predictor.joblib  # Modelo treinado
├── notebooks/             # Jupyter notebooks
├── src/
│   ├── __init__.py
│   ├── data_collector.py  # Coleta de dados da API
│   ├── data_processor.py  # Processamento de dados
│   ├── model.py          # Implementação do modelo
│   └── utils.py          # Funções utilitárias
├── streamlit_app/
│   └── app.py            # Interface do Streamlit
├── venv/                 # Ambiente virtual
├── .env                  # Variáveis de ambiente
├── .gitignore           # Arquivos ignorados pelo git
├── README.md            # Este arquivo
└── requirements.txt     # Dependências do projeto
```

## 🚀 Instalação e Uso

### Pré-requisitos
- Python 3.10 ou superior
- pip (gerenciador de pacotes Python)
- Git

### Instalação
1. Clone o repositório e entre no diretório:
```bash
git clone https://github.com/ClaudioAMF1/football_prediction.git
cd football_prediction
```

2. Crie e ative o ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
- Crie um arquivo `.env` na raiz do projeto
- Adicione sua chave da API:
```env
FOOTBALL_API_KEY=sua_chave_aqui
```

### Execução
1. Ative o ambiente virtual (se ainda não estiver ativo):
```bash
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

2. Execute o aplicativo:
```bash
streamlit run streamlit_app/app.py
```

## 📊 Modelo de Machine Learning
- **Algoritmo**: Random Forest Classifier
- **Features**: 
  - Média de gols
  - Aproveitamento
  - Forma recente
  - Confrontos diretos
- **Performance**:
  - Acurácia: ~55%
  - Superior ao baseline (33%)

## ⚠️ Limitações Conhecidas
- Quantidade limitada de dados da temporada atual
- Ausência de fatores externos (lesões, clima)
- Imprevisibilidade inerente do futebol

## 📝 Licença
Este projeto está sob a licença MIT.

## 👥 Autor
- Claudio Meireles e Kelwin Menezes
