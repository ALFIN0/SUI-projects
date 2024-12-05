# %% [markdown]
# Vítejte u prvního projektu do SUI.
# V rámci projektu Vás čeká několik cvičení, v nichž budete doplňovat poměrně malé fragmenty kódu (místo je vyznačeno pomocí `None` nebo `pass`).
# Pokud se v buňce s kódem již něco nachází, využijte/neničte to.
# Buňky nerušte ani nepřidávejte.
# Snažte se programovat hezky, ale jediná skutečně aktivně zakázaná, vyhledávaná a -- i opakovaně -- postihovaná technika je cyklení přes data (ať už explicitním cyklem nebo v rámci `list`/`dict` comprehension), tomu se vyhýbejte jako čert kříží a řešte to pomocí vhodných operací lineární algebry.
# 
# Až budete s řešením hotovi, vyexportujte ho ("Download as") jako PDF i pythonovský skript a ty odevzdejte **pojmenované názvem týmu** (tj. loginem vedoucího).
# Dbejte, aby bylo v PDF všechno vidět (nezůstal kód za okrajem stránky apod.).
# 
# U všech cvičení je uveden orientační počet řádků řešení.
# Berte ho prosím opravdu jako orientační, pozornost mu věnujte, pouze pokud ho významně překračujete.

# %%
import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy.stats

# %% [markdown]
# # Přípravné práce
# Prvním úkolem v tomto projektu je načíst data, s nimiž budete pracovat.
# Vybudujte jednoduchou třídu, která se umí zkonstruovat z cesty k negativním a pozitivním příkladům, a bude poskytovat:
# - pozitivní a negativní příklady (`dataset.pos`, `dataset.neg` o rozměrech [N, 7])
# - všechny příklady a odpovídající třídy (`dataset.xs` o rozměru [N, 7], `dataset.targets` o rozměru [N])
# 
# K načítání dat doporučujeme využít `np.loadtxt()`.
# Netrapte se se zapouzdřováním a gettery, berte třídu jako Plain Old Data.
# 
# Načtěte trénovací (`{positives,negatives}.trn`), validační (`{positives,negatives}.val`) a testovací (`{positives,negatives}.tst`) dataset, pojmenujte je po řadě `train_dataset`, `val_dataset` a `test_dataset`. 
# 
# **(6 řádků)** 
# 

# %%
class BinaryDataset:
    def __init__(self, positives_path, negatives_path):
        # nacitani pos a neg
        self.pos = np.loadtxt(positives_path)
        self.neg = np.loadtxt(negatives_path)
        # "stohovani" pozitivnich a negativnich
        self.xs = np.vstack((self.pos, self.neg))
        #targets jako bin hodnoty, konkatenace poctu kladnych a zapornych vysledku
        self.targets = np.concatenate((np.ones(self.pos.shape[0]), np.zeros(self.neg.shape[0])))
    pass

train_dataset = BinaryDataset('positives.trn', 'negatives.trn')
val_dataset = BinaryDataset('positives.val', 'negatives.val')
test_dataset = BinaryDataset('positives.tst', 'negatives.tst')

print('positives', train_dataset.pos.shape)
print('negatives', train_dataset.neg.shape)
print('xs', train_dataset.xs.shape)
print('targets', train_dataset.targets.shape)

# %% [markdown]
# V řadě následujících cvičení budete pracovat s jedním konkrétním příznakem. Naimplementujte proto funkci, která vykreslí histogram rozložení pozitivních a negativních příkladů z jedné sady. Nezapomeňte na legendu, ať je v grafu jasné, které jsou které. Funkci zavoláte dvakrát, vykreslete histogram příznaku `5` -- tzn. šestého ze sedmi -- pro trénovací a validační data
# 
# **(5 řádků)**

# %%
FOI = 5  # Feature Of Interest

def plot_data(poss, negs):
    #positives
    plt.hist(poss,label="Pozitivni",color="green",alpha=0.7,edgecolor='black',linewidth=0.6)
    #negatives
    plt.hist(negs,label="Negativni",color="red",alpha=0.7,edgecolor='black',linewidth=0.6)
    plt.xlabel('Hodnota')
    plt.ylabel('Cetnost')
    plt.legend()
    plt.show()
    pass
plot_data(train_dataset.pos[:, FOI], train_dataset.neg[:, FOI])
plot_data(val_dataset.pos[:, FOI], val_dataset.neg[:, FOI])



# %% [markdown]
# ### Evaluace klasifikátorů
# Než přistoupíte k tvorbě jednotlivých klasifikátorů, vytvořte funkci pro jejich vyhodnocování.
# Nechť se jmenuje `evaluate` a přijímá po řadě klasifikátor, pole dat (o rozměrech [N, F]) a pole tříd ([N]).
# Jejím výstupem bude _přesnost_ (accuracy), tzn. podíl správně klasifikovaných příkladů.
# 
# Předpokládejte, že klasifikátor poskytuje metodu `.prob_class_1(data)`, která vrací pole posteriorních pravděpodobností třídy 1 pro daná data.
# Evaluační funkce bude muset provést tvrdé prahování (na hodnotě 0.5) těchto pravděpodobností a srovnání získaných rozhodnutí s referenčními třídami.
# Využijte fakt, že `numpy`ovská pole lze mj. porovnávat se skalárem.
# 
# **(3 řádky)**

# %%
def evaluate(classifier, inputs, targets):
    post_prob_array = classifier.prob_class_1(inputs)
    # prahovani 
    threshold = post_prob_array > 0.5
    return np.mean(threshold == targets) #array s hodnotami evaluace zprumerovany pro uspesnost 


class Dummy:
    def prob_class_1(self, xs):
        return np.asarray([0.2, 0.7, 0.7])

print(evaluate(Dummy(), None, np.asarray([0, 0, 1])))  # should be 0.66

# %% [markdown]
# ### Baseline
# Vytvořte klasifikátor, který ignoruje vstupní data.
# Jenom v konstruktoru dostane třídu, kterou má dávat jako tip pro libovolný vstup.
# Nezapomeňte, že jeho metoda `.prob_class_1(data)` musí vracet pole správné velikosti.
# 
# **(4 řádky)**

# %%
class PriorClassifier:
    def __init__(self,class1):
        self.class1 = class1
    def prob_class_1(self,inputs):
        #naplneni pole hodnotami 
        return np.full(inputs.shape,self.class1)
    pass

baseline = PriorClassifier(0)
val_acc = evaluate(baseline, val_dataset.xs[:, FOI], val_dataset.targets)
print('Baseline val acc:', val_acc)

# %% [markdown]
# # Generativní klasifikátory
# V této  části vytvoříte dva generativní klasifikátory, oba založené na Gaussovu rozložení pravděpodobnosti.
# 
# Začněte implementací funce, která pro daná 1-D data vrátí Maximum Likelihood odhad střední hodnoty a směrodatné odchylky Gaussova rozložení, které data modeluje.
# Funkci využijte pro natrénovaní dvou modelů: pozitivních a negativních příkladů.
# Získané parametry -- tzn. střední hodnoty a směrodatné odchylky -- vypíšete.
# 
# **(1 řádek)**

# %%
def mle_gauss_1d(data):
    #tvar gaussovky urcen stredni hodnotou a smerodatnou odchylkou
    return np.mean(data),np.std(data) 

mu_pos, std_pos = mle_gauss_1d(train_dataset.pos[:, FOI])
mu_neg, std_neg = mle_gauss_1d(train_dataset.neg[:, FOI])

print('Pos mean: {:.2f} std: {:.2f}'.format(mu_pos, std_pos))
print('Neg mean: {:.2f} std: {:.2f}'.format(mu_neg, std_neg))

# %% [markdown]
# Ze získaných parametrů vytvořte `scipy`ovská gaussovská rozložení `scipy.stats.norm`.
# S využitím jejich metody `.pdf()` vytvořte graf, v němž srovnáte skutečné a modelové rozložení pozitivních a negativních příkladů.
# Rozsah x-ové osy volte od -0.5 do 1.5 (využijte `np.linspace`) a u volání `plt.hist()` nezapomeňte nastavit `density=True`, aby byl histogram normalizovaný a dal se srovnávat s modelem.
# 
# **(2 + 8 řádků)**

# %%
pos = scipy.stats.norm(loc=mu_pos, scale=std_pos)
neg = scipy.stats.norm(loc=mu_neg, scale=std_neg)

x_axis = np.linspace(-0.5, 1.5)
plt.hist(val_dataset.pos[:, FOI], density=True, color='green', alpha=0.7, edgecolor='black', linewidth=0.6, label='Pozitivni data')
plt.hist(val_dataset.neg[:, FOI], density=True, color='red', alpha=0.7, edgecolor='black', linewidth=0.6, label='Negativni data')

plt.plot(x_axis, pos.pdf(x_axis), label='Pozitivni model',color='green')
plt.plot(x_axis, neg.pdf(x_axis), label='Negativni model', color='red')

plt.title('Data vs Model')
plt.xlabel(f'Hodnota')
plt.ylabel('Hustota pravdepodobnosti')
plt.legend()

plt.show()

# %% [markdown]
# Naimplementujte binární generativní klasifikátor. 
# Při konstrukci přijímá dvě rozložení poskytující metodu `.pdf()` a odpovídající apriorní pravděpodobnost tříd.
# Dbejte, aby Vám uživatel nemohl zadat neplatné apriorní pravděpodobnosti.
# Jako všechny klasifikátory v tomto projektu poskytuje metodu `prob_class_1()`.
# 
# **(9 řádků)**

# %%
class GenerativeClassifier2Class:
    def __init__(self, dist1, dist0, aprior1, aprior0) -> None:
        #kontrola validity vstupu
        if aprior1 + aprior0 != 1 or aprior1 < 0 or aprior0 < 0 or aprior0 + aprior1 < 0:
            raise ValueError
        self.aprior1 = aprior1 #pozitivni priklady
        self.aprior0 = aprior0 #negativni priklady
        self.dist1 = dist1
        self.dist0 = dist0
        pass

    def prob_class_1(self, xs):
        #rozlozeni pravdepodobnosti jednotlivych trid
        p_1 = self.dist1.pdf(xs) 
        p_0 = self.dist0.pdf(xs)
        # bayesuv vzorec 
        return (p_1 * self.aprior1) / ((p_1 * self.aprior1) + (p_0 * self.aprior0))


    pass

# %% [markdown]
# Nainstancujte dva generativní klasifikátory: jeden s rovnoměrnými priory a jeden s apriorní pravděpodobností 0.75 pro třídu 0 (negativní příklady).
# Pomocí funkce `evaluate()` vyhodnotíte jejich úspěšnost na validačních datech.
# 
# **(2 řádky)**

# %%
classifier_flat_prior = GenerativeClassifier2Class(pos,neg,0.5,0.5)
classifier_full_prior = GenerativeClassifier2Class(pos,neg,0.25,0.75)

print('flat:', evaluate(classifier_flat_prior, val_dataset.xs[:, FOI], val_dataset.targets))
print('full:', evaluate(classifier_full_prior, val_dataset.xs[:, FOI], val_dataset.targets))

# %% [markdown]
# Vykreslete průběh posteriorní pravděpodobnosti třídy 1 jako funkci příznaku 5, opět v rozsahu <-0.5; 1.5> pro oba klasifikátory.
# Do grafu zakreslete i histogramy rozložení trénovacích dat, opět s `density=True` pro zachování dynamického rozsahu.
# 
# **(8 řádků)**

# %%
plt.hist(train_dataset.pos[:, FOI], density=True, color='green', alpha=0.7, edgecolor='black', linewidth=0.6, label='Pozitivni train data')
plt.hist(train_dataset.neg[:, FOI], density=True, color='red', alpha=0.7, edgecolor='black', linewidth=0.6, label='Negativni train data')

plt.plot(x_axis, classifier_flat_prior.prob_class_1(x_axis), label='Flat prior',color='green')
plt.plot(x_axis, classifier_full_prior.prob_class_1(x_axis), label='Full prior', color='red')

plt.xlabel(f'FOI: {FOI}')
plt.ylabel('Hodnota')
plt.legend()

plt.show()
pass

# %% [markdown]
# # Diskriminativní klasifikátory
# V následující části budete pomocí (lineární) logistické regrese přímo modelovat posteriorní pravděpodobnost třídy 1.
# Modely budou založeny čistě na NumPy, takže nemusíte instalovat nic dalšího.
# Nabitějších toolkitů se dočkáte ve třetím projektu.

# %%
def logistic_sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def binary_cross_entropy(probs, targets):
    return np.sum(-targets * np.log(probs) - (1-targets)*np.log(1-probs)) 

class LogisticRegressionNumpy:
    def __init__(self, dim):
        self.w = np.array([0.0] * dim)
        self.b = np.array([0.0])
        
    def prob_class_1(self, x):
        return logistic_sigmoid(x @ self.w + self.b)

# %% [markdown]
# Diskriminativní klasifikátor očekává, že dostane vstup ve tvaru `[N, F]`.
# Pro práci na jediném příznaku bude tedy zapotřebí vyřezávat příslušná data v správném formátu (`[N, 1]`). 
# Doimplementujte třídu `FeatureCutter` tak, aby to zařizovalo volání její instance.
# Který příznak se použije, nechť je konfigurováno při konstrukci.
# 
# Může se Vám hodit `np.newaxis`.
# 
# **(2 řádky)**

# %%
class FeatureCutter:
    def __init__(self, fea_id):
        self.fea_id = fea_id
        pass
        
    def __call__(self, x):
        return x[:,self.fea_id,np.newaxis]

# %% [markdown]
# Dalším krokem je implementovat funkci, která model vytvoří a natrénuje.
# Jejím výstupem bude (1) natrénovaný model, (2) průběh trénovací loss a (3) průběh validační přesnosti.
# Neuvažujte žádné minibatche, aktualizujte váhy vždy na celém trénovacím datasetu.
# Po každém kroku vyhodnoťte model na validačních datech.
# Jako model vracejte ten, který dosáhne nejlepší validační přesnosti.
# Jako loss použijte binární cross-entropii  a logujte průměr na vzorek.
# Pro výpočet validační přesnosti využijte funkci `evaluate()`.
# Oba průběhy vracejte jako obyčejné seznamy.
# 
# **(cca 11 řádků)**

# %%
def train_logistic_regression(nb_epochs, lr, in_dim, fea_preprocessor):
    model = LogisticRegressionNumpy(in_dim)  
    best_model = copy.deepcopy(model)  
    losses = []
    accuracies = []
    
    
    train_X = fea_preprocessor(train_dataset.xs)
    train_t = train_dataset.targets
    best_acc = 0
    for _ in range(nb_epochs):
        y = model.prob_class_1(train_X)
        #trenovaci loss
        loss = (binary_cross_entropy(y,train_t)) / len(train_t)
        losses.append(loss)
        #trenink regrese
        model.w = model.w  - lr * (y-train_t).dot(train_X) 
        model.b = model.b - lr * np.sum(y - train_t)
        
        # validacni presnost
        val_X = fea_preprocessor(val_dataset.xs)
        val_acc = evaluate(model, val_X, val_dataset.targets)
        accuracies.append(val_acc)
        
        # aktualizace modelu
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)
    
    return best_model, losses, accuracies


# %% [markdown]
# Funkci zavolejte a natrénujte model.
# Uveďte zde parametry, které vám dají slušný výsledek.
# Měli byste dostat přesnost srovnatelnou s generativním klasifikátorem s nastavenými priory.
# Neměli byste potřebovat víc, než 100 epoch.
# Vykreslete průběh trénovací loss a validační přesnosti, osu x značte v epochách.
# 
# V druhém grafu vykreslete histogramy trénovacích dat a pravděpodobnost třídy 1 pro x od -0.5 do 1.5, podobně jako výše u generativních klasifikátorů.
# 
# **(1 + 5 + 8 řádků)**

# %%
disc_fea5, losses, accuracies = train_logistic_regression(100,0.001,1,FeatureCutter(FOI))

pass
print('w', disc_fea5.w.item(), 'b', disc_fea5.b.item())

pass
print('disc_fea5:', evaluate(disc_fea5, val_dataset.xs[:, FOI][:, np.newaxis], val_dataset.targets))



# Trenovaci loss
plt.figure(figsize=(10,5)) # pro lepsi citelnost 2 grafu
plt.subplot(1, 2, 1)
plt.plot(range(0,100), losses, label='Trenovaci Loss', color='red')
plt.xlabel('Iterace')
plt.ylabel('Loss')
# Validacni presnost
plt.subplot(1, 2, 2)
plt.plot(range(0,100), accuracies, label='Validacni presnost', color='green')
plt.xlabel('Iterace')
plt.ylabel('Presnost')
plt.show()
#rozhodl jsem se umistit do 2 grafu, kvuli prehlednosti

# Vykreslení histogramů a pravděpodobnosti třídy 1
probs_class1 = disc_fea5.prob_class_1(x_axis[:, np.newaxis])

plt.hist(train_dataset.pos[:, FOI], density=True, color='green', alpha=0.7, edgecolor='black', linewidth=0.6, label='Pozitivni')
plt.hist(train_dataset.neg[:, FOI], density=True, color='red', alpha=0.7, edgecolor='black', linewidth=0.6, label='Negativni')
plt.plot(x_axis, probs_class1, label='Pravdepodobnost tridy 1', color='green', linestyle='-')

# Popisky grafu
plt.title('Trenovaci data a pravdepodobnost tridy 1')
plt.xlabel(f'FOI: {FOI}')
plt.ylabel('Hustota / Pravdepodobnost')
plt.legend()

plt.show()

# %% [markdown]
# ## Všechny vstupní příznaky
# V posledním cvičení natrénujete logistickou regresi, která využije všechn sedm vstupních příznaků.
# Zavolejte funkci z předchozího cvičení, opět vykreslete průběh trénovací loss a validační přesnosti.
# Měli byste se dostat nad 90 % přesnosti.
# 
# Může se Vám hodit `lambda` funkce.
# 
# **(1 + 5 řádků)**

# %%
disc_full_fea, losses, accuracies = train_logistic_regression(100, 0.001, 7, lambda x:x)

# plot trenovaci loss a validacni presnost
# trenovaci loss a accuracies mi neplotuje nektere hodnoty, evaluace regrese nabyva vsak validni? hodnoty

# presnost modelu
print('disc_full_fea:', evaluate(disc_full_fea, val_dataset.xs, val_dataset.targets))


# %% [markdown]
# # Závěrem
# Konečně vyhodnoťte všech pět vytvořených klasifikátorů na testovacích datech.
# Stačí doplnit jejich názvy a předat jim odpovídající příznaky.
# Nezapomeňte, že u logistické regrese musíte zopakovat formátovací krok z `FeatureCutter`u.

# %%
xs_full = test_dataset.xs
xs_foi = test_dataset.xs[:, FOI]
targets = test_dataset.targets

print('Baseline:', evaluate(baseline, xs_foi, targets))
print('Generative classifier (w/o prior):', evaluate(classifier_flat_prior, xs_foi, targets))
print('Generative classifier (correct):', evaluate(classifier_full_prior, xs_foi, targets))
print('Logistic regression:', evaluate(disc_fea5, xs_foi[:, np.newaxis], targets))
print('logistic regression all features:', evaluate(disc_full_fea, xs_full, targets))

# %% [markdown]
# Blahopřejeme ke zvládnutí projektu! Nezapomeňte (1) spustit celý notebook načisto (Kernel -> Restart & Run all), (2) zkontrolovat, že všechny výpočty prošly podle očekávání, a (3) před odevzdáním pojmenovat soubory loginem vedoucího týmu.
# 
# Mimochodem, vstupní data nejsou synteticky generovaná.
# Nasbírali jsme je z baseline řešení historicky prvního SUI projektu; vaše klasifikátory v tomto projektu predikují, že daný hráč vyhraje dicewars, takže by se daly použít jako heuristika pro ohodnocování listových uzlů ve stavovém prostoru hry.
# Pro představu, data jsou z pozic pět kol před koncem partie pro daného hráče.
# Poskytnuté příznaky popisují globální charakteristiky stavu hry jako je například poměr délky hranic předmětného hráče k ostatním hranicím.
# Nejeden projekt v ročníku 2020 realizoval požadované "strojové učení" v agentovi hrajicím dicewars kopií domácí úlohy, která byla předchůdkyní tohoto projektu.

# %%



