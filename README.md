# IS 577 Data Mining: text mining of Alaska environmental observations

Final project for IS 577 (Data Mining), Spring 2025, University of Illinois Urbana-Champaign.

## What the project does

The [Local Environmental Observer (LEO) Network](https://www.leonetwork.org) is a platform where people post about unusual animal, environment and weather events. Each post is labelled by hand with one or more topics, such as `FISH`, `PERMAFROST` or `WILDFIRE`.

This project takes the posts for Alaska (3,568 entries, 105 topic labels) and asks two questions:

1. **What do the posts say?** Exploratory analysis of the text (most common words and word pairs, word counts), sentiment analysis (polarity and subjectivity with NLTK and TextBlob) and clustering of the summaries.
2. **Can the topic labels be assigned automatically?** Multi-label text classification with logistic regression, multinomial Naive Bayes, random forest and a stacked model that combines the three.

The slides with the results are in `IS577_leo_emine2_jhmyers2_presentation.pdf`.

## Files

| File or folder | What it holds |
| --- | --- |
| `leo.ipynb` | Sentiment analysis and maps |
| `eda.ipynb` | Exploratory analysis of labels and text |
| `clusters_00.ipynb`, `clusters_01.ipynb` | Clustering of post summaries |
| `logi_reg.ipynb`, `model_01.ipynb`, `model_01.py` | Classification models |
| `csv-arff.ipynb` | Converts the data to ARFF format for Weka |
| `leo_search_results.csv` | Posts downloaded from the LEO Network (Alaska) |
| `leo_summary.csv`, `leo_summary_cleaned.csv`, `clustered_summaries.csv` | Processed versions of the data |
| `figures/`, `pol_vs_sub.png` | Figures used in the slides |
| `html_codes/` | HTML exports of the notebooks |
| `sentiment_choropleth_map_alaska.html` | Interactive sentiment map of Alaska |
| `requirements.txt` | Python packages used |

## Data source

Post titles, summaries and topic labels come from the LEO Network, <https://www.leonetwork.org>.
