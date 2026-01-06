# House Prices - Regularized Linear Regression Pipeline (Ridge/Lasso/ElasticNet)

## Proje Özeti
Bu çalışma, konut fiyatlarını (**SalePrice**) tahmin etmek için uçtan uca bir **regresyon pipeline**’ı kurar.  
Amaç; veri sızıntısı (leakage) riskini minimize eden, kategorik/numerik özellikleri doğru işleyen, düzenlileştirme (regularization) ve model teşhisi (residual diagnostics) ile desteklenen **sağlam** bir model elde etmektir.

## Hedef Değişken
- **Target:** `SalePrice`

## Veri Hazırlama ve Kararlar

### 1) Kimlik alanı çıkarımı
- `Id` bir kimlik alanı olduğu için modelden çıkarılmıştır.

### 2) Yüksek eksik oranlı sütunların temizlenmesi
Eksik oranı çok yüksek olan değişkenler modelleme açısından anlamlı sinyal taşımaktan ziyade gürültü üretme riskine sahiptir.  
Bu nedenle **%80 ve üzeri eksik** değer oranına sahip değişkenler veri setinden çıkarılmıştır:

- `WallMat`
- `PoolQC`
- `MiscFeature`
- `Alley`
- `Fence`

> Not: Eksik oranı hesaplaması “tam steril” olmak için `X_train` üzerinde yapılıp, ardından hem train hem test setine aynı sütun düşümü uygulanmıştır.

### 3) Özellik tipleri (numeric / categorical)
- Numerik sütunlar: `int64`, `float64`
- Kategorik sütunlar: `object`

Pipeline içinde dönüşümler:
- **Numerik:** `median` ile imputasyon
- **Kategorik:** `most_frequent` ile imputasyon + `OneHotEncoder(handle_unknown="ignore")`

### 4) Nadir kategori kontrolü
Bazı kategorik sütunlarda çok düşük frekanslı (nadir) sınıflar gözlemlenmiştir (ör. tekil değerler).  
Bu, model kararlılığı ve genellenebilirlik açısından not edilmiştir.

## Modelleme Yaklaşımı

### Baseline: Ridge Regression
İlk aşamada Ridge ile baseline kurulmuştur.  
Test metrikleri (örnek çıktı):
- **RMSE:** ~5221
- **MAE:** ~1307
- **R²:** ~0.996

> RMSE ve MAE, sklearn sürüm uyumluluğu için `mean_squared_error(..., squared=False)` yerine `np.sqrt(mean_squared_error(...))` yaklaşımıyla hesaplanmıştır.

### Cross-Validation (Genellenebilirlik)
Modelin genellenebilirliğini görmek için **5-fold cross-validation** uygulanmıştır.  
Ridge / Lasso / ElasticNet karşılaştırması yapılmıştır.

Örnek özet (scaled preprocessing ile):
- **Ridge mean R²:** ~0.973 (std ~0.013)
- **Lasso mean R²:** ~0.972 (std ~0.010)
- **ElasticNet mean R²:** ~0.971 (std ~0.013)

#### ConvergenceWarning Notu (Lasso/ElasticNet)
Lasso/ElasticNet eğitiminde zaman zaman `ConvergenceWarning` görülebilir.
Bu uyarı genellikle:
- iterasyon sayısının yetersiz olması
- özelliklerin ölçeklenmemesi
- alpha/l1_ratio değerlerinin uygunsuzluğu
nedenleriyle oluşur.

Bu nedenle ikinci karşılaştırmada:
- numerik değişkenlere **StandardScaler**
- `max_iter` artırımı
uygulanmıştır.

## Feature Importance & Feature Selection
Permutation importance ile değişken katkıları incelenmiştir.  
Örnek çıktılarda `Pesos` değişkeninin öneminin anormal derecede yüksek olması, veri kaynağı/feature anlamı açısından ayrıca doğrulanmalıdır.

Ayrıca düşük katkılı bazı değişkenler tespit edilmiştir:
- Örn. `BsmtHalfBath`, `Street`, `ChimneyStyle`, `LandContour`, `LandSlope`, `Utilities`, `RoofMatl`, `Condition2`, vb.

Feature selection sonrası performans karşılaştırması yapıldığında skor farkının ihmal edilebilir düzeyde olması, çıkarılan değişkenlerin modele anlamlı katkı sağlamadığını göstermiştir.

## Residual (Hata) Analizi
Modelin hata davranışını incelemek için:
- Gerçek vs Tahmin grafiği (y=x referans çizgisi ile)
- Residual vs Tahmin (homoskedastisite kontrolü)
- Residual dağılımı (hist + KDE)
- QQ-Plot (normallik kontrolü)
grafikleri üretilmiştir.

Gözlem:
- Tahminler genel olarak referans çizgisine çok yakındır.
- Residual grafiğinde **belirgin bir aykırı gözlem** dikkat çekmektedir.
- QQ-Plot, özellikle uçlarda normalden sapma olabileceğini işaret etmektedir (outlier etkisi muhtemel).

## Klasör Yapısı (Bu çalışma için)
Bu README, aşağıdaki dizinde tutulur:

- `code/totkanligizem/03-Decision-Science/olist/`
  - `Bonus.ipynb` (çalışma not defteri)
  - `totkanligizem.README.md` (bu dosya)

## Nasıl Çalıştırılır (Özet)
- Notebook’u aç: `Bonus.ipynb`
- Hücreleri sırayla çalıştır:
  - veri hazırlama
  - preprocessing + model
  - cross-validation
  - feature selection
  - residual analizi

## Sonuç
Kurulan pipeline:
- leakage riskini azaltan şekilde tasarlanmış,
- kategorik/numerik dönüşümleri güvenli biçimde ele alan,
- regularization ve CV ile genellenebilirliği test edilmiş,
- residual analizi ile model davranışı teşhis edilmiş
uçtan uca bir regresyon çözümüdür.

