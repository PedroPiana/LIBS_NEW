# Documentacao do modulo `src`

Este arquivo resume o que existe dentro de `src/`, como cada modulo funciona e como usar as funcoes e classes principais no projeto.

## Visao geral

O pacote `src` concentra as rotinas reutilizaveis do projeto:

- configuracao de caminhos e diretorios
- leitura e escrita de arquivos
- preprocessamento de espectros
- modelos classicos e customizados
- treino e avaliacao de pipelines
- metricas e graficos

Em notebooks e scripts, o uso normal e:

```python
from src.config import DATA_DIR, MODELS_DIR, ensure_dirs
from src.preprocessing import make_snv, make_savgol
from src.models import PLSDAClassifier
from src.training import run_experiment
```

## Como o pacote funciona

O arquivo `src/__init__.py` transforma a pasta `src` em um pacote Python. Isso permite importar os modulos com `from src import ...` ou `from src.modulo import ...`.

O fluxo tipico do projeto e:

1. Ler os dados com `src.io` ou pandas.
2. Montar preprocessamentos com `src.preprocessing`.
3. Escolher modelos com `src.models`.
4. Executar pipelines com `src.training`.
5. Salvar metricas, JSONs e graficos com `src.evaluation` e `src.plotting`.

## `src/config.py`

Define caminhos base e constantes globais.

### `BASE_DIR`

Raiz do projeto. E calculada a partir da localizacao do proprio arquivo `config.py`.

### `DATA_DIR`

Caminho para a pasta `dataset/`.

### `MODELS_DIR`

Caminho para a pasta `models/`.

### `PLOTS_DIR`

Caminho para a pasta `plots/`.

### `MODELS_GROUPKFOLD_DIR`

Caminho para a pasta `models_groupkfold/`.

### `RANDOM_SEED`

Seed padrao para processos aleatorios.

### `LASERS`

Tupla com os lasers suportados: `266`, `532` e `1064`.

### `ensure_dirs(dirs=None)`

Garante que os diretorios importantes existam.

Como funciona:

- se `dirs` for `None`, cria os diretorios padrao do projeto
- se `dirs` for informado, cria apenas os caminhos fornecidos

Uso:

```python
from src.config import ensure_dirs

ensure_dirs()
```

## `src/utils.py`

Funcoes auxiliares genericas para labels, JSON e caminhos.

### `extrair_numero(label)`

Extrai o primeiro numero encontrado em uma string.

Como funciona:

- recebe uma string como `"A12"`
- busca o primeiro trecho numerico com expressao regular
- retorna `12`
- se nao houver numero, retorna `None`

Uso:

```python
from src.utils import extrair_numero

extrair_numero("A12")
```

### `normalize_missing_values(value)`

Converte estruturas do NumPy para tipos nativos do Python para facilitar salvamento em JSON.

Como funciona:

- percorre recursivamente `dict`, `list`, `tuple` e `set`
- converte `numpy.ndarray` para lista
- converte `numpy.generic` para valor nativo

Uso normal:

```python
from src.utils import normalize_missing_values
```

### `ensure_parent_dir(path)`

Garante que a pasta pai do arquivo exista antes de salvar.

Como funciona:

- recebe um caminho de arquivo
- cria o diretório pai com `parents=True`
- retorna o `Path` final

### `save_json(output_path, payload)`

Salva qualquer estrutura serializavel em JSON.

Como funciona:

- cria a pasta de destino se necessario
- normaliza valores do NumPy
- grava o JSON com indentacao e UTF-8

### `load_json(input_path)`

Carrega e retorna o conteudo de um JSON.

### `save_pipeline_summary_json(output_path, *args)`

Salva um resumo de pipeline em JSON e aceita dois formatos.

Como funciona:

- com 1 argumento, recebe diretamente o `payload` completo
- com 7 argumentos, aceita o formato legado usado em notebooks
- em ambos os casos, chama `save_json`

Isso existe para manter compatibilidade com scripts antigos e com a nova estrutura.

## `src/preprocessing.py`

Transformadores para usar em `sklearn.Pipeline`.

### `SavgolFilter`

Implementa o filtro Savitzky-Golay como transformador scikit-learn.

Como funciona:

- `fit` nao treina nada e apenas retorna `self`
- `transform` aplica `scipy.signal.savgol_filter` ao longo do eixo 1
- e util para suavizar espectros

Parametros:

- `window_length`: tamanho da janela
- `polyorder`: ordem do polinomio
- `deriv`: derivada a calcular

Uso:

```python
from src.preprocessing import SavgolFilter

filtro = SavgolFilter(window_length=11, polyorder=2)
```

### `SNV`

Implementa Standard Normal Variate.

Como funciona:

- calcula media e desvio padrao de cada amostra individualmente
- centraliza e normaliza cada espectro
- desvio padrao zero e tratado como 1 para evitar divisao por zero

Uso:

```python
from src.preprocessing import SNV
```

### `make_savgol(window_length=11, polyorder=2)`

Cria uma etapa nomeada `("savgol", SavgolFilter(...))` para pipelines.

### `make_snv()`

Cria a etapa nomeada `("snv", SNV())`.

### `make_scaler()`

Cria a etapa nomeada `("scaler", StandardScaler())`.

Essas funcoes sao usadas para montar pipelines com nomes consistentes.

## `src/models.py`

Contem modelos classicos e wrappers customizados.

### `TENSORFLOW_AVAILABLE`

Indica se TensorFlow foi importado com sucesso no ambiente atual.

### `PLSDAClassifier`

Implementacao binaria de PLS-DA baseada em `PLSRegression`.

Como funciona:

- `fit` codifica as classes com `LabelEncoder`
- exige exatamente 2 classes
- ajusta um `PLSRegression`
- `predict` gera scores e usa limiar `0.5`
- converte o resultado de volta para os rótulos originais

Uso:

```python
from src.models import PLSDAClassifier

model = PLSDAClassifier(n_components=2)
```

### `PLSDAMulticlass`

Versao multiclasse de PLS-DA.

Como funciona:

- usa `LabelBinarizer` para codificar as classes
- treina um `PLSRegression` por classe
- `predict` escolhe a classe com maior score
- `predict_proba` normaliza os scores positivos por linha

### `CNN1DClassifier`

Wrapper scikit-learn para uma CNN 1D com Keras.

Como funciona:

- so funciona se TensorFlow estiver instalado
- `fit` converte `y` com `LabelEncoder`
- exige 2 classes
- adiciona uma dimensao extra em `X` para formato `(amostras, canais)`
- cria a rede em `_build_model`
- usa `EarlyStopping` para parar cedo quando a validacao nao melhora
- `predict` retorna as classes originais

### `_build_model(input_length)`

Metodo interno de `CNN1DClassifier`.

Ele monta a arquitetura da rede:

- `InputLayer`
- duas camadas `Conv1D`
- `MaxPooling1D`
- `Flatten`
- `Dense`
- `Dropout`
- saida binaria com `sigmoid`

### `make_svm_linear()`

Cria um `SVC` com kernel linear.

### `make_svm_rbf()`

Cria um `SVC` com kernel RBF.

### `make_svm_poly()`

Cria um `SVC` com kernel polinomial.

### `make_svm_sigmoid()`

Cria um `SVC` com kernel sigmoid.

### `make_random_forest()`

Cria um `RandomForestClassifier` com `200` arvores e `random_state=42`.

### `build_binary_model_registry(include_cnn=False)`

Monta um dicionario com modelos binarios prontos para uso.

Como funciona:

- sempre inclui SVMs, Random Forest e PLS-DA
- inclui `cnn_1d` apenas se `include_cnn=True` e TensorFlow estiver disponivel

### `build_multiclass_model_registry(include_cnn=False)`

Monta um registro de modelos para multiclasse.

Hoje ele inclui:

- `pls_da`
- opcionalmente `cnn_1d`

## `src/training.py`

Organiza a montagem e avaliacao de pipelines.

### `build_pipeline(preprocess_steps, model)`

Cria um `sklearn.pipeline.Pipeline` com preprocessamento + modelo.

Como funciona:

- recebe uma lista de etapas preprocessadoras nomeadas
- adiciona a etapa final `("model", model)`

### `evaluate_pipeline(pipe, X, y, cv, groups=None, scoring="accuracy")`

Executa validacao cruzada e retorna scores e predicoes.

Como funciona:

- usa `cross_val_score` para medir desempenho por fold
- usa `cross_val_predict` para gerar predicoes fora da amostra
- se `groups` for informado, passa os grupos para o validador

Retorna:

- `scores`: array com os scores por fold
- `y_pred`: predicoes agregadas da validacao cruzada

### `save_trained_pipeline(pipe, output_path)`

Salva o pipeline treinado com `joblib`.

Como funciona:

- tenta importar `joblib`
- se nao estiver instalado, levanta `ImportError`
- grava o objeto no caminho informado

### `run_experiment(...)`

Executa a grade de experimentos do projeto.

Como funciona:

- cria as pastas de saida se nao existirem
- combina cada preprocessamento com cada modelo
- avalia cada pipeline com `evaluate_pipeline`
- tenta treinar o pipeline final e salvar o modelo treinado
- tenta salvar a matriz de confusao
- monta resumos por pipeline
- salva um resumo consolidado em JSON
- gera o ranking final dos pipelines

Parametros principais:

- `X`, `y`: dados e rotulos
- `preprocessamentos`: dicionario com etapas de preprocessamento
- `modelos`: dicionario com estimadores
- `cv`: objeto de validacao cruzada
- `output_models_dir`: pasta para modelos treinados
- `output_plots_dir`: pasta para graficos e resumos
- `laser`: identificador do laser usado
- `labels`: classes usadas no resumo
- `experiment_name`: nome do experimento
- `display_labels`: labels exibidos na matriz de confusao
- `groups`: grupos para `GroupKFold` ou similares

Retorna um dicionario com:

- `results`
- `pipeline_summaries`
- `summary_path`
- `ranking_path`

## `src/evaluation.py`

Funcoes para metricas e resumos de avaliacao.

### `compute_confusion_matrix(y_true, y_pred)`

Retorna a matriz de confusao usando `sklearn.metrics.confusion_matrix`.

### `save_confusion_matrix_plot(y_true, y_pred, output_path, title, display_labels)`

Cria e salva um grafico de matriz de confusao.

Como funciona:

- calcula a matriz de confusao
- monta um `ConfusionMatrixDisplay`
- salva a figura em alta resolucao

### `build_pipeline_summary(pipeline_name, scores, mean_accuracy, std_accuracy, laser, labels, n_splits)`

Monta um resumo padrao de um pipeline.

O retorno inclui:

- nome do pipeline
- laser
- labels
- numero de folds
- scores por fold
- media e desvio padrao

### `save_pipeline_summary(output_path, summary)`

Salva um resumo de pipeline em JSON.

Internamente chama `save_pipeline_summary_json` de `src.utils`.

### `save_consolidated_summary(output_path, experiment, laser, labels, n_splits, pipeline_summaries)`

Salva um resumo consolidado do experimento.

Como funciona:

- monta um JSON com metadados gerais do experimento
- inclui a lista de resumos de cada pipeline

### `plot_ranking(results, output_path, title, xlabel="Pipeline", ylabel="Accuracy")`

Cria o grafico de ranking dos pipelines.

Como funciona:

- recebe um dicionario `nome -> score`
- plota barras com os valores
- anota os valores no grafico
- salva o resultado em arquivo

Observacao: este modulo tem funcoes com nomes parecidos com `src.plotting.py`. Na pratica, `evaluation.py` foca no fluxo de avaliacao e resumos, enquanto `plotting.py` oferece utilitarios visuais mais gerais.

## `src/plotting.py`

Funcoes visuais gerais para graficos do projeto.

### `save_figure(fig, output_path)`

Salva uma figura Matplotlib em disco.

Como funciona:

- garante que a pasta pai exista
- salva com `bbox_inches="tight"`
- fecha a figura para liberar memoria

### `save_confusion_matrix_plot(y_true, y_pred, output_path, title=None, display_labels=None)`

Cria uma matriz de confusao usando `seaborn.heatmap`.

Como funciona:

- calcula a matriz de confusao
- plota o heatmap com anotacoes
- ajusta eixos e titulo se informados
- salva com `save_figure`

### `plot_ranking(results, output_path, title=None)`

Plota o ranking de scores ordenado do maior para o menor.

Como funciona:

- extrai nomes e scores do dicionario
- ordena por score decrescente
- cria um grafico de barras horizontais com `seaborn`
- salva a figura em disco

### `plot_spectra(X, wavelengths=None, labels=None, output_path=None, title=None)`

Plota espectros individuais ou em lote.

Como funciona:

- converte `X` para `numpy.ndarray`
- se `wavelengths` nao for informado, usa indices de coluna
- se `X` for 1D, plota um unico espectro
- se `X` for 2D, plota cada espectro da matriz
- se `output_path` existir, salva a figura; caso contrario, retorna o objeto `fig`

## `src/io.py`

Utilitarios de leitura e persistencia de arquivos.

### `ensure_parent_dir(path)`

Cria a pasta pai do caminho e retorna o `Path` original.

### `read_csv(path, **kwargs)`

Atalho para `pandas.read_csv`.

### `save_json(obj, output_path)`

Salva um objeto em JSON com indentacao.

### `load_json(input_path)`

Carrega um JSON e retorna o objeto decodificado.

### `save_joblib(obj, output_path)`

Salva qualquer objeto com `joblib.dump`.

### `load_joblib(path)`

Carrega um objeto salvo com joblib.

### `save_keras_model(model, output_path, save_format=None)`

Salva um modelo Keras/TensorFlow.

Como funciona:

- tenta importar TensorFlow/Keras
- se falhar, levanta `ImportError`
- salva o modelo no formato indicado pelo sufixo ou por `save_format`

### `load_keras_model(path)`

Carrega um modelo Keras salvo em disco.

## `src/__init__.py`

Nao contem logica de processamento. Ele apenas expõe os modulos do pacote para facilitar imports.

## Como usar na pratica

### Exemplo de pipeline simples

```python
from sklearn.svm import SVC
from src.preprocessing import make_snv
from src.training import build_pipeline

pipe = build_pipeline([make_snv()], SVC(kernel="linear"))
```

### Exemplo com avaliacao

```python
from src.models import PLSDAClassifier
from src.training import run_experiment

results = run_experiment(
    X=X,
    y=y,
    preprocessamentos={"none": []},
    modelos={"pls_da": PLSDAClassifier(n_components=2)},
    cv=sgkf,
    output_models_dir="models_groupkfold/laser532",
    output_plots_dir="plots_groupkfold/laser532",
    laser=532,
    labels=["A", "R"],
    experiment_name="demo",
    display_labels=["Arabica", "Robusta"],
    groups=groups,
)
```

## Observacoes uteis

- `src.models` tem wrappers customizados para integrar modelos ao ecossistema scikit-learn.
- `src.training.run_experiment` e o ponto central para comparar varios pipelines de forma automatizada.
- Existem funcoes de graficos em `src.evaluation` e em `src.plotting`; se precisar de uma API mais geral, use `src.plotting`.
- Os caminhos padrao sao pensados para a estrutura atual do workspace, com pastas como `dataset/`, `models/` e `plots/` na raiz.
