# Student Performance Prediction — WEKA & Python

Predicting students' final grades from academic and behavioural features, using three
classification algorithms. The project was first built in **WEKA**, then rebuilt in
**Python (pandas + scikit-learn)** with proper cross-validation.

The rebuild produced an unexpected result, and that result is the main finding of this project.

مشروع للتنبؤ بالدرجة النهائية للطالب باستخدام ثلاث خوارزميات تصنيف. نُفّذ أولاً في WEKA، ثم أُعيد بناؤه في Python مع تقييم صحيح — والنتيجة كانت اكتشاف مشكلة جوهرية في البيانات نفسها.

---

## TL;DR

The dataset does not support the task. Rows 11–1000 were randomly generated and contain no
relationship between the features and the target. Under stratified 10-fold cross-validation all
three models score at chance level (~10%). The originally reported 51.59% does not reproduce
under held-out evaluation — it reflects memorisation, not learning.

The full audit, evidence, and re-evaluation are in
[`student_performance_analysis.ipynb`](student_performance_analysis.ipynb).

---

## Dataset

`student_performance_updated_1000.csv` — 1,000 rows, 12 columns.

| Column | Type | Notes |
|---|---|---|
| `StudentID` | numeric | Identifier — dropped before training |
| `Name` | text | Identifier, ~962 unique values — dropped before training |
| `Gender` | categorical | Male / Female |
| `AttendanceRate` | numeric | 0–95 |
| `StudyHoursPerWeek` | numeric | 0–30 |
| `PreviousGrade` | numeric | 0–90 |
| `ExtracurricularActivities` | numeric | 0–3 |
| `ParentalSupport` | categorical | Low / Medium / High |
| **`FinalGrade`** | **target** | 10 distinct values — treated as a nominal class |
| `Study Hours` | numeric | Duplicate concept of `StudyHoursPerWeek`, inconsistent |
| `Attendance (%)` | numeric | Duplicate concept of `AttendanceRate`, inconsistent |
| `Online Classes Taken` | boolean | True / False |

Because `FinalGrade` has 10 classes, random guessing scores about **10%** and the majority-class
baseline is **11.46%**. Any meaningful model must clearly beat those.

---

## Methodology

### Phase 1 — WEKA

- Preprocessing: missing-value handling, normalisation, attribute removal
- Models: J48 decision tree, Naive Bayes, Random Forest
- Outputs: model summaries, ROC curves, comparison chart
- Reported best result: Random Forest at **51.59%**

### Phase 2 — Python rebuild

The same pipeline in code, with two additions the WEKA run did not include: a full data quality
audit before modelling, and stratified 10-fold cross-validation for every model.

| WEKA | scikit-learn |
|---|---|
| J48 | `DecisionTreeClassifier` |
| NaiveBayes | `GaussianNB` |
| RandomForest | `RandomForestClassifier` |

---

## Data quality audit

| Issue | Evidence |
|---|---|
| Missing values in every column | 22–50 per column (~3–5%) |
| Duplicate columns that disagree | `AttendanceRate` vs `Attendance (%)` → r ≈ 0 |
| Impossible values | Attendance recorded up to **200%** |
| Identifier columns present as features | `Name` has ~962 unique values across 1,000 rows |
| **No feature correlates with the target** | All correlations ≈ 0; the strongest is `StudentID` |

---

## Key finding

Plotting `PreviousGrade` against `FinalGrade` separates the dataset into two very different parts:

| Subset | Correlation with `FinalGrade` |
|---|---|
| Rows 1–10 | **0.998** |
| Rows 11–1000 | **−0.005** |

The first ten rows are a small, hand-written, realistic sample. The remaining 990 rows are random
noise appended to pad the file to 1,000 records. No algorithm can extract a genuine pattern from
them.

---

## Results

Stratified 10-fold cross-validation, after cleaning:

| Model | CV accuracy | Std dev | Training accuracy |
|---|---|---|---|
| Decision Tree (J48) | **11.04%** | ±3.46 | 100.00% |
| Naive Bayes | **9.90%** | ±2.99 | 16.46% |
| Random Forest | **10.52%** | ±2.40 | 100.00% |
| *Majority-class baseline* | *11.46%* | — | — |
| *Random guessing* | *10.00%* | — | — |

### Why the original 51.59% did not reproduce

Both tree models reach **100% accuracy on the training data** and collapse to **~10% on unseen
data**. That gap is textbook overfitting: with no real signal, a decision tree still fits the
data perfectly by splitting it down to single-row leaves, and the memorised tree is worthless on
anything new.

The 51.59% falls between those two figures, which is consistent with an evaluation that was not
fully held out. Under cross-validation it disappears — correctly, because there was never a
pattern to find.

---

## What this project demonstrates

The valuable outcome here is the diagnosis, not an accuracy number. Reporting a high score from
an unvalidated run would have concealed a dataset that cannot support its own task. Detecting
that — and proving it with correlation analysis, baseline comparison, and held-out evaluation —
is the more useful skill.

---

## Repository structure

```
machine-learning-student-performance/
│
├── README.md
├── student_performance_analysis.ipynb      # Python rebuild — audit, models, findings
├── student_performance_updated_1000.csv    # Dataset
├── Student Performance Prediction.pdf      # Original project report
│
└── weka/                                   # Phase 1 outputs
    ├── Preprocessing steps missing value.PNG
    ├── Preprocessing steps normalize.PNG
    ├── Preprocessing steps remove.PNG
    ├── Model J48.PNG
    ├── Model NaiveBays.PNG
    ├── Model Random Forest.PNG
    ├── Visualization ROC J48.PNG
    ├── Visualization ROC NaiveBays.PNG
    ├── Visualization ROC Random Forest.PNG
    └── bar chart.png
```

> The screenshots currently sit in the repository root. Moving them into a `weka/` folder keeps
> the root readable — update the paths above if you keep them where they are.

---

## Running the notebook

```bash
git clone https://github.com/lolo67fa/machine-learning-student-performance.git
cd machine-learning-student-performance

pip install pandas numpy matplotlib scikit-learn
jupyter notebook student_performance_analysis.ipynb
```

The notebook is committed with its outputs, so GitHub renders every chart and result without
running anything.

---

## Next step

Rerun the same pipeline on the **UCI Student Performance** dataset (Cortez & Silva, 2008) —
real records from two Portuguese secondary schools, with genuine predictive structure:

<https://archive.ics.uci.edu/dataset/320/student+performance>

Every cell after the cleaning section works unchanged once `DATA_PATH` points at the new file.
The comparison — chance level on synthetic data, real performance on real data — is a stronger
result than either run on its own.

---

## Author

**Ghala Alshreef**

- GitHub: [@lolo67fa](https://github.com/lolo67fa)
- LinkedIn: [ghala-a-670a62380](https://linkedin.com/in/ghala-a-670a62380)
- Portfolio: [try.ka.nz/ai/ghalaalshreef](https://try.ka.nz/ai/ghalaalshreef)

---

## License

Released under the MIT License.
