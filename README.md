# SomosNLP Hackathon 2025 - Gastronomía Hispana

Proyecto de fine-tuning de modelos de lenguaje especializados en gastronomía hispana utilizando técnicas de QLoRA y DPO (Direct Preference Optimization).

## Descripción del Proyecto

Este proyecto desarrolla modelos de lenguaje especializados en recetas y cultura gastronómica hispana, con enfoque particular en la cocina ecuatoriana y colombiana. Utiliza técnicas avanzadas de fine-tuning como QLoRA y DPO para crear asistentes culinarios que comprenden tanto las técnicas de cocina como el contexto cultural de los platos tradicionales.

## Estructura del Proyecto

```
├── notebooks/
│   ├── dpo.ipynb              # Entrenamiento con DPO (Direct Preference Optimization)
│   └── qlora.ipynb            # Fine-tuning con QLoRA
├── scripts/
│   ├── analyze_dataset.py      # Análisis y filtrado de datasets por longitud de tokens
│   ├── convert_json_to_parquet.py  # Conversión de JSON a formato Parquet
│   ├── dataset-cohere-dpo.py  # Generación de datasets DPO con Cohere API
│   ├── esbieta.py             # Scraper de recetas de recetasdesbieta.com
│   ├── question_bank.py       # Generación de preguntas sobre recetas
│   ├── youtube_count.py       # Extracción de estadísticas de videos de YouTube
│   └── yt_transcript.py       # Extracción de transcripciones de YouTube
└── README.md
```

## Configuración Inicial

### Prerrequisitos

- Python 3.8+
- CUDA compatible GPU (recomendado para entrenamiento)
- Tokens de API:
  - Cohere API key
  - Hugging Face token
  - Weights & Biases account (opcional)


## 📊 Datos y Datasets

### Fuentes de Datos

- **Recetas web**: Scraping automatizado de sitios especializados
- **Videos de YouTube**: Transcripciones y metadatos de videos culinarios
- **Preguntas generadas**: Banco de preguntas educativas sobre recetas

### Datasets Generados

- `patrimonio-gastronomico-hispano`: Dataset principal de recetas
- `gastronomia-hispana-dpo`: Dataset de pares de preferencias para DPO

## 🤖 Entrenamiento de Modelos

### QLoRA Fine-tuning

**Notebook**: `notebooks/qlora.ipynb`

Entrenamiento supervisado para generar respuestas sobre recetas tradicionales:

```python
# Configuración del modelo base
MODEL_BASE = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
MAX_SEQ_LENGTH = 4096

# Parámetros de LoRA
LORA_R = 32
LORA_ALPHA = 32
LEARNING_RATE = 2e-4
```

**Características**:
- ✅ Optimización de memoria con cuantización 4-bit
- ✅ Template ChatML para conversaciones
- ✅ Evaluación continua durante entrenamiento
- ✅ Guardado automático de checkpoints

### DPO Training

**Notebook**: `notebooks/dpo.ipynb`

Optimización de preferencias para mejorar la calidad de respuestas:

```python
# Configuración DPO
MODEL_BASE = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
BETA = 0.8  # Parámetro de control de preferencias
LEARNING_RATE = 2e-7
```

**Características**:
- ✅ Entrenamiento con pares chosen/rejected
- ✅ Control fino de preferencias culturales
- ✅ Validación de calidad de respuestas
- ✅ Métricas de evaluación especializadas

## 🛠️ Scripts de Utilidad

### Recolección de Datos

```bash
# Scraping de recetas
python scripts/esbieta.py --output recetas.json --max-recipes 1000

# Extracción de transcripciones de YouTube
python scripts/yt_transcript.py

# Obtención de estadísticas de videos
python scripts/youtube_count.py
```

### Generación de Datasets

```bash
# Generar banco de preguntas
python scripts/question_bank.py

# Crear dataset DPO
python scripts/dataset-cohere-dpo.py

# Convertir formatos
python scripts/convert_json_to_parquet.py
```

### Análisis de Datos

```bash
# Analizar longitud de secuencias
python scripts/analyze_dataset.py
```

## 📈 Modelos Entrenados

### Mistral-7B Gastronomía Hispana
- **Base**: Mistral-7B-Instruct-v0.3
- **Especialización**: Recetas ecuatorianas y colombianas
- **Técnica**: QLoRA fine-tuning
- **Formato**: ChatML

### Qwen3-8B Gastronomía Hispana
- **Base**: Qwen3-8B
- **Especialización**: Optimización de preferencias culinarias
- **Técnica**: DPO training
- **Formato**: ChatML

## 🎭 Características de los Modelos

### Capacidades Especializadas

- **Conocimiento Cultural**: Comprende el contexto histórico y regional de los platos
- **Técnicas Culinarias**: Explica métodos de cocción tradicionales y modernos
- **Ingredientes Locales**: Conoce ingredientes específicos de cada región
- **Adaptación de Porciones**: Ajusta recetas según el número de comensales
- **Solución de Problemas**: Ayuda a resolver errores comunes en la cocina

### Formatos Soportados

- **Conversación Natural**: Interacción fluida en español
- **Recetas Paso a Paso**: Instrucciones detalladas y claras
- **Consejos Técnicos**: Tips profesionales de cocina
- **Contexto Cultural**: Historia y tradiciones gastronómicas

## Evaluación y Métricas

### Métricas de Entrenamiento

- **Loss de Entrenamiento**: Monitoreado con Weights & Biases
- **Perplexity**: Medida de fluidez del modelo
- **BLEU Score**: Calidad de generación de texto
- **Validación Cultural**: Revisión manual de autenticidad

### Pruebas de Calidad

```python
# Ejemplo de evaluación
messages = [
    {"role": "user", "content": "¿Cómo preparar encebollado ecuatoriano?"}
]

# El modelo debe responder con:
# - Ingredientes específicos del plato
# - Técnicas tradicionales
# - Contexto cultural apropiado
# - Instrucciones claras y precisas
```

## 📊 Resultados y Benchmarks

### Métricas de Rendimiento

| Modelo | Parámetros | VRAM Utilizada | Tiempo Entrenamiento |
|--------|------------|----------------|---------------------|
| Mistral-7B | 7B | ~16GB | ~2 horas |
| Qwen3-8B | 8B | ~18GB | ~3 horas |

### Casos de Uso

- ✅ **Asistente Culinario**: Respuestas sobre recetas tradicionales
- ✅ **Educación Gastronómica**: Enseñanza de técnicas y cultura
- ✅ **Preservación Cultural**: Documentación de tradiciones culinarias
- ✅ **Adaptación Regional**: Personalización según ubicación geográfica

## 🤝 Contribuciones

### Cómo Contribuir

1. **Fork el repositorio**
2. **Crear una rama de feature**: `git checkout -b feature/nueva-funcionalidad`
3. **Commit cambios**: `git commit -am 'Agregar nueva funcionalidad'`
4. **Push a la rama**: `git push origin feature/nueva-funcionalidad`
5. **Crear Pull Request**

### Áreas de Contribución

- 🍳 **Nuevas Recetas**: Ampliar la base de datos con más platos regionales
- 🌎 **Nuevas Regiones**: Incluir gastronomía de otros países hispanos
- 🤖 **Mejoras de Modelo**: Optimizaciones en arquitectura y entrenamiento
- 📱 **Interfaces**: Desarrollo de aplicaciones y demos interactivas

## 📜 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- **SomosNLP**: Por organizar el hackathon y proporcionar la plataforma
- **Unsloth**: Por las optimizaciones de entrenamiento eficiente
- **Cohere**: Por proporcionar acceso a su API para generación de datos
- **Comunidad Open Source**: Por las herramientas y librerías utilizadas

## Contacto

- **Equipo**: somosnlp-hackathon-2025
- **Repository**: [GitHub](https://github.com/somosnlp-hackathon-2025)
- **Modelos**: [Hugging Face](https://huggingface.co/somosnlp-hackathon-2025)

---

**¡Preservemos y compartamos la riqueza de la gastronomía hispana a través de la inteligencia artificial!** 🇪🇨 🥘