---
editor_options: 
  markdown: 
    wrap: 72
---

# Annexe technique : préliminaires et preuves

> Annexe du [document compagnon](00b_synthetic_methods.md) du notebook
> `00b_synthetic_methods.ipynb` (projet `01-ts-uncertainty`). Version du
> 2026-11-18. Les étiquettes entre crochets ([GR07], [KB78]…) renvoient
> au §9 du document compagnon, avec leurs statuts. Un résultat classique
> dont la source n'a pas été relue est signalé « non vérifié ». Les
> calculs signalés « vérifié par simulation » ont été contrôlés
> numériquement le 2026-11-18 ; ceux qui ne figurent pas dans le
> notebook l'ont été dans un script de rédaction, hors dépôt.

**Partie I — Préliminaires** (notions utilisées, avec un exemple chiffré
chaque fois que c'est possible) [P1 stationnarité, ergodicité](#P1) ·
[P2 risque et répétitions](#P2) · [P3 filtres, causalité, inversibilité
; l'oracle](#P3) · [P4 chaînes de Markov](#P4) · [P5 espace d'état et
filtre de Kalman](#P5) · [P6 volatilité stochastique](#P6) · [P7 modèles
à seuil](#P7) · [P8 processus gaussiens](#P8) · [P9 EDS](#P9) · [P10
tests de racine unitaire](#P10) · [P11 Newey–West](#P11) · [P12 conforme
séparé et glissant](#P12) · [P13 calibration](#P13) · [P14 protocole
sans fuite](#P14)

**Partie II — Preuves longues** : [T1 causal et inversible](#T1) · [T2
mémoire courte et mémoire longue](#T2) · [T3 moments du GARCH](#T3) ·
[T4 variance de la pente et variance de long terme](#T4) · [T5
régression fallacieuse](#T5)

------------------------------------------------------------------------

# Partie I — Préliminaires

<a id="P1"></a> \## P1. Stationnarité au second ordre, ergodicité

**Définition.** Un processus $(X_t)_{t\in\mathbb Z}$ est *stationnaire
au second ordre* (ou faiblement stationnaire) si (i)
$\mathbb E X_t^2<\infty$, (ii) $\mathbb E X_t=m$ ne dépend pas de $t$,
(iii) $\operatorname{Cov}(X_t,X_{t+h})=\gamma(h)$ ne dépend que du
décalage $h$, pas de $t$. Oui : « la covariance ne dépend que du lag »,
avec en plus une moyenne constante et une variance finie. On note
$\gamma(0)=\operatorname{Var}X_t$ et $\rho(h)=\gamma(h)/\gamma(0)$. La
stationnarité *stricte* demande l'invariance par translation de toutes
les lois fini-dimensionnelles ; pour un processus gaussien, les deux
notions coïncident.

**Exemples.** Bruit blanc : $\gamma(h)=\sigma^2\mathbf 1_{h=0}$. AR(1)
$X_t=\varphi X_{t-1}+\varepsilon_t$, $|\varphi|<1$ :
$\gamma(h)=\sigma^2\varphi^{|h|}/(1-\varphi^2)$. Marche aléatoire
$S_t=\sum_{j\le t}\varepsilon_j$ : $\operatorname{Var}S_t=t\sigma^2$
dépend de $t$, donc non stationnaire. Une moyenne $\mu_t$ non constante
(tendance, saison) rend $X=\mu+S$ non stationnaire même si $S$ l'est :
d'où la décomposition (H2) du document compagnon.

**Ergodicité.** Un processus stationnaire est *ergodique* si les
moyennes temporelles sur une trajectoire convergent vers les espérances
:
$n^{-1}\sum_{t\le n}g(X_t,\dots,X_{t+k})\to\mathbb E\,g(X_0,\dots,X_k)$
presque sûrement (théorème ergodique de Birkhoff ; non vérifié ici). Les
ARMA à bruit i.i.d. sont ergodiques. L'ergodicité donne la convergence,
pas la vitesse : l'écart-type d'une moyenne temporelle est de l'ordre de
$\sqrt{\omega^2/n}$ avec $\omega^2=\sum_h\gamma(h)$ la variance de long
terme (T4), plus grande que $\gamma(0)$ quand la corrélation est
positive.

<a id="P2"></a> \## P2. Risque, moyenne empirique, et pourquoi $R$
répétitions

**Deux espérances à distinguer.** Une *procédure* $\mathcal A$
transforme un bloc d'apprentissage $D$ en un prédicteur $\hat f_D$,
fonction de l'historique $X_{\le t}$. Avec une perte $\ell$ :

-   le risque *conditionnel à l'apprentissage*
    $R(\hat f_D)=\mathbb E\big[\ell(X_{t+h},\hat f_D(X_{\le t}))\mid D\big]$
    : espérance sur l'origine de test et sur le futur, le prédicteur
    étant fixé ;
-   le risque *de la procédure* $\bar R=\mathbb E_D\,R(\hat f_D)$ : on
    moyenne en plus sur l'aléa de l'apprentissage. C'est lui qui compare
    les méthodes.

**Exemple 1 : rien à apprendre.** Persistance $\hat f(X_{\le t})=X_t$,
perte quadratique, AR(1) stationnaire :
$R=\mathbb E(X_{t+1}-X_t)^2=2\gamma(0)(1-\rho(1))=2\gamma(0)(1-\varphi)$.
L'espérance porte sur la loi stationnaire du couple $(X_t,X_{t+1})$ ;
aucun aléa d'apprentissage.

**Exemple 2 : un paramètre appris.** $\hat f_D(X_t)=\hat\varphi X_t$,
$\hat\varphi$ estimé sur $n_0$ points. Comme
$X_{t+1}=\varphi X_t+\varepsilon_{t+1}$ avec $\varepsilon_{t+1}$
indépendant de $(X_t,D)$ :
$$R(\hat f_D)=\sigma^2+\gamma(0)(\hat\varphi-\varphi)^2 .$$ Il dépend de
$D$ par $\hat\varphi$. Or
$\mathbb E(\hat\varphi-\varphi)^2\approx(1-\varphi^2)/n_0$ (variance
asymptotique de l'estimateur des moindres carrés d'un AR(1), non
démontrée ici) et $\gamma(0)(1-\varphi^2)=\sigma^2$, donc
$\bar R\approx\sigma^2(1+1/n_0)$ : estimer un paramètre coûte une
fraction $1/n_0$ du risque irréductible.

**Moyenne empirique.** $\hat R_n=n^{-1}\sum_{i\le n}\ell_i$ sur les $n$
origines du bloc de test. Pour un prédicteur *figé* (paramètres déjà
estimés sur $D$, indépendants des données de test) et un processus
stationnaire ergodique, le théorème ergodique donne
$\hat R_n\to R(\hat f_D)$ : une seule longue trajectoire suffit pour le
risque conditionnel de *ce* prédicteur. Si les paramètres étaient
réestimés sur le bloc de test, la moyenne serait biaisée vers le bas
(fuite).

**Pourquoi répéter sur** $R$ trajectoires indépendantes ? 1. $\hat R_n$
ne voit qu'*un* tirage de l'erreur d'estimation $\hat\varphi-\varphi$ ;
or on compare des procédures, c'est-à-dire $\bar R$. 2. Les $\ell_i$
sont dépendantes : $\operatorname{Var}\hat R_n\approx\omega_\ell^2/n$,
où $\omega_\ell^2=\sum_h\operatorname{Cov}(\ell_0,\ell_h)$ est la
variance de long terme de la perte (T4). L'erreur-type naïve
$\hat s/\sqrt n$ la sous-estime d'un facteur de l'ordre de $\sqrt{r}$,
avec $r=\omega_\ell^2/\operatorname{Var}\ell_0$ (Idée 8). 3. $R$
trajectoires indépendantes donnent $R$ valeurs indépendantes de
$\hat R_n$ et donc une erreur-type valide $\hat s/\sqrt R$, sans
modéliser la dépendance.

Le choix $R=20$ est limité par le temps de calcul ; un écart inférieur à
deux erreurs-types n'est pas interprétable.

<a id="P3"></a> \## P3. Filtres linéaires, causalité, inversibilité ;
l'oracle n'inverse pas de matrice

**Filtre linéaire.** Avec l'opérateur retard $B$ ($(BX)_t=X_{t-1}$), un
filtre transforme une suite d'entrée $\varepsilon$ en
$Y_t=\sum_jc_j\varepsilon_{t-j}=c(B)\varepsilon_t$, avec
$c(z)=\sum_jc_jz^j$ (fonction de transfert) et $\sum|c_j|<\infty$. Il
est *causal* si $c_j=0$ pour $j<0$ : la sortie à l'instant $t$ ne dépend
que de l'entrée jusqu'à $t$, jamais de l'avenir.

**ARMA.** $\varphi(B)X_t=\theta(B)\varepsilon_t$ s'écrit
$X=\psi(B)\varepsilon$ avec $\psi(z)=\theta(z)/\varphi(z)$ : $X$ est un
filtre de son bruit. Il est causal si $\varphi$ n'a pas de racine dans
$|z|\le1$. Il est *inversible* si $\theta$ n'a pas de racine dans
$|z|\le1$ : alors $\varepsilon=\pi(B)X$, $\pi=\varphi/\theta$,
c'est-à-dire que le bruit est un filtre causal des observations (le mot
vient de l'inversion de l'opérateur $\theta(B)$ en série de $B$).
Preuves : T1.

**Pourquoi les deux conditions.** La causalité garantit que $X_t$ ne
dépend que du passé du bruit. L'inversibilité garantit que
$\varepsilon_t$ se calcule à partir des observations jusqu'à $t$ : il
est donc $\mathcal F_t$-mesurable (c'est l'*innovation*, l'erreur de
prévision à un pas), ce qui rend légitime la formule de prévision de
l'Idée 2.

**L'oracle inverse-t-il une matrice de covariance ?** Non. Pour un $S$
gaussien centré de covariance $\Sigma$, la meilleure prévision s'écrit
en toute généralité (conditionnement gaussien)
$$\mathbb E[S_{t+h}\mid S_{1:t}]=\Sigma_{t+h,1:t}\,\Sigma_{1:t}^{-1}\,S_{1:t},$$
avec une matrice $t\times t$ à inverser (coût $O(t^3)$). Pour un
ARMA(1,1), la même quantité se calcule par la récurrence
$$\hat\varepsilon_s=S_s-\varphi S_{s-1}-\theta\hat\varepsilon_{s-1},\qquad \mathbb E[S_{t+h}\mid\mathcal F_t]=\varphi^{h-1}(\varphi S_t+\theta\hat\varepsilon_t),$$
en $O(t)$. Une erreur d'initialisation sur $\hat\varepsilon$ est
multipliée par $-\theta$ à chaque pas : elle meurt en $|\theta|^t$,
c'est l'inversibilité (T1). Dans E1, l'oracle n'a même pas besoin de la
récurrence : il lit la vraie innovation $\varepsilon_t$ du générateur
(cellule : `PHI**(h-1) * (PHI*S_t + THETA*e_t)`), c'est-à-dire la limite
$t\to\infty$ de la récurrence. Pour un modèle non markovien (processus
gaussien général, P8), c'est la formule matricielle ou le filtre de
Kalman (P5) qui fait le calcul.

<a id="P4"></a> \## P4. Chaînes de Markov : vocabulaire, états latents,
filtre

Une chaîne à $K$ états a pour matrice de transition
$P_{ik}=\mathbb P(s_{t+1}=k\mid s_t=i)$ (lignes de somme 1).

-   *Homogène* : $P$ ne dépend pas de $t$.
-   *Irréductible* : de tout état on atteint tout autre état, en un
    nombre fini de pas, avec une probabilité positive.
-   *Apériodique* : le plus grand commun diviseur des durées de retour
    possibles à un état est 1 (il suffit qu'un état ait $P_{ii}>0$).

Théorème classique (non vérifié ici) : une chaîne finie, homogène,
irréductible et apériodique a une unique loi stationnaire $\pi$
($\pi P=\pi$) et $P^h\to\mathbf 1\pi$ à vitesse géométrique
$|\lambda_2|^h$.

**Exemple du notebook** : deux états, probabilité de rester $0.95$ :
$P=\begin{pmatrix}0.95&0.05\\0.05&0.95\end{pmatrix}$, valeurs propres
$1$ et $\lambda_2=0.9$, $\pi=(\tfrac12,\tfrac12)$,
$P^h=\tfrac12\mathbf 1\mathbf 1^\top+\tfrac12\,0.9^h\begin{pmatrix}1&-1\\-1&1\end{pmatrix}$,
durée moyenne de séjour $1/(1-0.95)=20$ pas.

**États latents, « sachant la chaîne ».** La chaîne $(s_t)$ n'est pas
observée (*latente*). On observe $X_t$, dont la loi dépend de $s_t$. «
Sachant la chaîne » signifie *conditionnellement à toute la trajectoire
des états latents* : les $X_t$ sont alors indépendants,
$X_t\mid s_t=k\sim N(\mu_k,\sigma_k^2)$. Marginalement (en intégrant la
chaîne), les $X_t$ sont dépendants : toute la dépendance temporelle
vient de la chaîne cachée.

**Filtre de Hamilton** (forme standard ; [Ham89], filtre non relu). Les
probabilités filtrées $\pi_{t|t}(k)=\mathbb P(s_t=k\mid X_{1:t})$
vérifient $\pi_{t|t}(k)\propto\pi_{t|t-1}(k)\,f_k(X_t)$ ($f_k$ : densité
de la composante $k$) puis $\pi_{t+1|t}=\pi_{t|t}P$ : une règle de Bayes
suivie d'une propagation. La loi prédictive à $h$ pas est le mélange
$\sum_k(\pi_{t|t}P^h)_k\,N(\mu_k,\sigma_k^2)$ (loi totale).

<a id="P5"></a> \## P5. Modèles d'espace d'état et filtre de Kalman
(niveau local)

**Le modèle.** Quand on écrit $X_t=\mu_t+S_t$ avec « $\mu_t$ marche
aléatoire », le plus simple est le *modèle de niveau local* :
$$X_t=\mu_t+\varepsilon_t\quad(\text{observation}),\qquad \mu_t=\mu_{t-1}+\eta_t\quad(\text{état latent}),$$
avec $\varepsilon_t\sim N(0,\sigma_\varepsilon^2)$ et
$\eta_t\sim N(0,\sigma_\eta^2)$ indépendants. Si $\sigma_\eta^2=0$, le
niveau est constant (tendance déterministe, ici nulle) ; si
$\sigma_\eta^2>0$, il dérive : tendance *aléatoire*. Le rapport
signal/bruit $q=\sigma_\eta^2/\sigma_\varepsilon^2$ mesure l'aléa de la
tendance. (Avec une pente $\beta_t=\beta_{t-1}+\zeta_t$ : tendance
linéaire locale ; le cas (D) du document compagnon est celui de
variances d'état nulles.)

**Le filtre.** On cherche la loi de $\mu_t$ sachant $X_{1:t}$ ; elle est
gaussienne, $N(\hat\mu_{t|t},P_{t|t})$. Récurrence :

-   prédiction : $\hat\mu_{t|t-1}=\hat\mu_{t-1|t-1}$,
    $P_{t|t-1}=P_{t-1|t-1}+\sigma_\eta^2$ ;
-   gain : $K_t=P_{t|t-1}/(P_{t|t-1}+\sigma_\varepsilon^2)\in(0,1)$ ;
-   mise à jour :
    $\hat\mu_{t|t}=\hat\mu_{t|t-1}+K_t(X_t-\hat\mu_{t|t-1})$,
    $P_{t|t}=(1-K_t)P_{t|t-1}$.

*Preuve.* Sachant $X_{1:t-1}$, le couple $(\mu_t,X_t)$ est gaussien de
moyenne $(m,m)$, avec $m=\hat\mu_{t|t-1}$, et de covariance
$\begin{pmatrix}P&P\\P&P+\sigma_\varepsilon^2\end{pmatrix}$,
$P=P_{t|t-1}$. La formule de conditionnement gaussien donne la moyenne
$m+\frac{P}{P+\sigma_\varepsilon^2}(X_t-m)$ et la variance
$P-\frac{P^2}{P+\sigma_\varepsilon^2}=(1-K_t)P$. ∎

**Régime permanent et lissage exponentiel.** $P_{t|t-1}\to P$, solution
de
$P=\frac{P\sigma_\varepsilon^2}{P+\sigma_\varepsilon^2}+\sigma_\eta^2$,
et $K_t\to K$. Alors $\hat\mu_{t|t}=(1-K)\hat\mu_{t-1|t-1}+KX_t$ :
moyenne mobile exponentielle, de poids $K(1-K)^j$. **Exemple**
$\sigma_\varepsilon^2=1$, $\sigma_\eta^2=0.5$ ($q=0.5$) :
$P^2-0.5P-0.5=0$ donne $P=1$ et $K=0.5$ (vérifié par simulation : le
gain vaut 0.5000 après 2000 pas).

**Lien avec les ARIMA.**
$\Delta X_t=\eta_t+\varepsilon_t-\varepsilon_{t-1}$ est un MA(1) :
$\gamma(0)=\sigma_\eta^2+2\sigma_\varepsilon^2$,
$\gamma(1)=-\sigma_\varepsilon^2$, $\gamma(h)=0$ pour $h\ge2$, donc
$\rho(1)=-1/(q+2)$ ($=-0.4$ pour $q=0.5$ ; vérifié par simulation :
$-0.400$). Le MA(1) $\Delta X_t=u_t+\theta u_{t-1}$ vérifie
$\theta/(1+\theta^2)=\rho(1)$ : deux solutions $\theta$ et $1/\theta$
(T1) ; on retient l'inversible, $|\theta|<1$ : $\theta=-0.5$ pour
$q=0.5$, soit $\theta=-(1-K)$ dans cet exemple. Le niveau local est donc
un ARIMA(0,1,1) et sa prévision est le lissage exponentiel (en général,
résultat classique ; non vérifié ici).

**Prévision.** $\mathbb E[X_{t+h}\mid X_{1:t}]=\hat\mu_{t|t}$ pour tout
$h$ ; variance d'erreur $P_{t|t}+h\sigma_\eta^2+\sigma_\varepsilon^2$ :
elle croît linéairement avec $h$ si $\sigma_\eta^2>0$ et reste bornée si
$\sigma_\eta^2=0$. C'est l'Idée 9 (borné ou croissant). La loi de
$\mu_t$ sachant les observations est le pendant gaussien du filtre de
Hamilton (P4) ; référence : [DK12] (ouvrage non relu).

<a id="P6"></a> \## P6. Volatilité stochastique, par contraste avec
GARCH

**Le modèle** (forme standard ; [KSC98] non relu) :
$$X_t=e^{h_t/2}z_t,\qquad h_t=\mu_h+\phi\,(h_{t-1}-\mu_h)+\sigma_\eta\,\eta_t,$$
avec $z_t,\eta_t$ i.i.d. $N(0,1)$ indépendants. $h_t=\log\sigma_t^2$ est
un AR(1) *latent*.

**Contraste avec GARCH.** Dans un GARCH, $\sigma_t^2$ est une fonction
déterministe du passé observé ($\mathcal F_{t-1}$-mesurable) : la
vraisemblance est le produit des densités conditionnelles, explicite.
Dans la volatilité stochastique, $h_t$ a son propre bruit $\eta_t$ : on
ne peut pas le reconstruire exactement. La vraisemblance est l'intégrale
en dimension $n$ $\int\prod_tp(X_t\mid h_t)\,p(h_{1:n})\,dh$, sans forme
close : on recourt au filtrage ou à des méthodes de Monte Carlo (c'est
l'objet de [KSC98], non relu). La variance conditionnelle n'est donc pas
exactement prévisible.

**Moments** (vérifiés par simulation). $h_t\sim N(\mu_h,\sigma_h^2)$
avec $\sigma_h^2=\sigma_\eta^2/(1-\phi^2)$. Comme
$\mathbb Ee^{sh}=\exp(s\mu_h+s^2\sigma_h^2/2)$ :
$\mathbb EX^2=e^{\mu_h+\sigma_h^2/2}$,
$\mathbb EX^4=3e^{2\mu_h+2\sigma_h^2}$ et le kurtosis vaut
$3e^{\sigma_h^2}>3$. Avec $\phi=0.95$, $\sigma_\eta=0.2$, $\mu_h=0$ :
$\sigma_h^2=0.410$, kurtosis théorique 4.522, observé 4.521.

<a id="P7"></a> \## P7. Modèles à seuil (TAR, STAR) : d'où vient le
seuil ?

**Idée.** Un modèle linéaire par régime, le régime étant fixé par une
variable *observée*. TAR à deux régimes :
$$X_t=\begin{cases}\varphi^{(1)}X_{t-1}+\varepsilon_t&\text{si }X_{t-d}\le r,\\ \varphi^{(2)}X_{t-1}+\varepsilon_t&\text{si }X_{t-d}>r.\end{cases}$$
$r$ est le *seuil*, $d\ge1$ le *délai*. Par contraste avec les régimes
de Markov (P4), le régime n'est pas latent : c'est une fonction de
$X_{t-d}$, donc la prévision à un pas est une simple règle.

**D'où vient le seuil.** C'est un paramètre libre, estimé. Pour $(r,d)$
fixés, chaque régime se régresse par moindres carrés ; on balaie $r$ sur
les quantiles de l'échantillon et on garde le couple $(r,d)$ qui
minimise la somme des carrés résiduels (vraisemblance profilée). La
fonction de $r$ est constante par morceaux : la recherche se fait sur
une grille, pas par gradient. STAR remplace l'indicatrice par une
transition lisse $G\big((X_{t-d}-r)/\gamma\big)$ (logistique, par
exemple) :
$X_t=\big[(1-G)\varphi^{(1)}+G\varphi^{(2)}\big]X_{t-1}+\varepsilon_t$.

**Pourquoi.** Un processus gaussien stationnaire est réversible dans le
temps et n'engendre pas de cycles limites non linéaires (résultat
classique ; non vérifié ici). Un modèle à seuil peut produire des
asymétries et des cycles non linéaires ; c'est le sujet de [TL80] (fiche
lue ; contenu non relu).

<a id="P8"></a> \## P8. Processus gaussiens et noyaux

**Définition.** $S\sim\mathrm{GP}(m,k)$ : pour tous $t_1,\dots,t_n$,
$(S_{t_i})_i$ est gaussien de moyenne $m(t_i)$ et de covariance
$k(t_i,t_j)$, où le noyau $k$ est symétrique défini positif. Le
processus est stationnaire si et seulement si $k(s,t)=k(s-t)$. **Le
noyau est la structure de dépendance** : choisir $k$, c'est choisir
$\gamma$.

**Exemple.** $k(\tau)=\frac{\sigma^2}{2\kappa}e^{-\kappa|\tau|}$ (noyau
exponentiel, Matérn $\nu=\tfrac12$ [RW06]) est la covariance de l'OU
(P9). Échantillonné au pas $\Delta$ :
$\gamma(j)=\gamma(0)\,e^{-\kappa\Delta|j|}=\gamma(0)\varphi^{|j|}$ avec
$\varphi=e^{-\kappa\Delta}$, c'est-à-dire *exactement* la covariance
d'un AR(1) (Idée 5).

**Prévision.** Pour $S_*$ à prédire et $S$ observé, de covariance $K$ :
$\mathbb E[S_*\mid S]=k_*^\top K^{-1}S$ et
$\operatorname{Var}(S_*\mid S)=k_{**}-k_*^\top K^{-1}k_*$
(conditionnement gaussien, comme en P3). Coût $O(n^3)$ ; pour un noyau
markovien comme l'exponentiel, le filtre de Kalman (P5) donne le même
résultat en $O(n)$.

<a id="P9"></a> \## P9. Équations différentielles stochastiques

$dX_t=a(X_t)\,dt+b(X_t)\,dW_t$ est la notation différentielle de
$X_t=X_0+\int_0^ta(X_s)\,ds+\int_0^tb(X_s)\,dW_s$ (intégrale d'Itô). $a$
est la *dérive* (tendance locale), $b$ la *diffusion* (amplitude du
bruit), $W$ le mouvement brownien ($W_0=0$, accroissements indépendants
de loi $N(0,\Delta)$).

-   **OU** : $a(x)=-\kappa x$, $b=\sigma$ (rappel vers 0). **Transition
    exacte** : par la formule d'Itô,
    $d(e^{\kappa t}S_t)=e^{\kappa t}\sigma\,dW_t$ (le terme
    $\kappa e^{\kappa t}S_t\,dt$ compense la dérive). Intégrée entre $t$
    et $t+\Delta$ :
    $S_{t+\Delta}=e^{-\kappa\Delta}S_t+\sigma\int_t^{t+\Delta}e^{-\kappa(t+\Delta-u)}dW_u$.
    L'intégrale de Wiener est gaussienne, centrée, indépendante de
    $\mathcal F_t$, de variance
    $\sigma^2\int_0^\Delta e^{-2\kappa s}ds=\sigma^2\frac{1-e^{-2\kappa\Delta}}{2\kappa}$
    (isométrie d'Itô). C'est un AR(1) de coefficient
    $\varphi=e^{-\kappa\Delta}$ ; sa variance stationnaire $v$ vérifie
    $v=\varphi^2v+\sigma^2\frac{1-\varphi^2}{2\kappa}$, d'où
    $v=\sigma^2/(2\kappa)$.
-   **CIR** : $a=\kappa(\theta-x)$, $b=\sigma\sqrt x$ : la variance du
    bruit est proportionnelle au niveau ; le processus reste positif si
    $2\kappa\theta\ge\sigma^2$ (condition de Feller ; non vérifié ici).
-   **Double puits** : $a(x)=x-x^3=-V'(x)$,
    $V(x)=\tfrac{x^4}4-\tfrac{x^2}2$.
-   **Euler–Maruyama** :
    $X_{k+1}=X_k+a(X_k)\Delta+b(X_k)\sqrt\Delta\,\xi_k$,
    $\xi_k\sim N(0,1)$ i.i.d. Pour l'OU, c'est un AR(1) de coefficient
    $c=1-\kappa\Delta$ et de variance d'innovation $\sigma^2\Delta$,
    donc de variance stationnaire
    $\sigma^2\Delta/(1-c^2)=\sigma^2/(\kappa(2-\kappa\Delta))$.
-   **Fokker–Planck** : la densité $p(x,t)$ vérifie
    $\partial_tp=-\partial_x(ap)+\tfrac12\partial_{xx}(b^2p)$. Une
    densité stationnaire à flux nul vérifie $\tfrac12(b^2p)'=ap$ ; pour
    $b=\sigma$ et $a=-V'$ : $p\propto\exp(-2V/\sigma^2)$.

<a id="P10"></a> \## P10. Tests de racine unitaire : à quoi servent ADF
et KPSS

**Question.** La série a-t-elle une *racine unitaire* (marche aléatoire

:   un choc est permanent) ou est-elle stationnaire (un choc s'efface),
    éventuellement autour d'une tendance ? La réponse commande la
    modélisation : différencier ou non, et la forme des intervalles à
    grand horizon (Idée 9).

-   **Dickey–Fuller** [DF79]. Modèle $x_t=\rho x_{t-1}+u_t$,
    $H_0:\rho=1$. On régresse $\Delta x_t=(\rho-1)x_{t-1}+u_t$ et on lit
    la statistique $t$ de $(\rho-1)$. Sous $H_0$ elle n'a ni la loi de
    Student ni la loi normale : sa loi limite est une fonctionnelle
    brownienne (du type de T5), tabulée (résumé lu : lois limites de
    l'estimateur et du test $t$ sous $\rho=1$).
-   **ADF** (*augmented* DF) [Said–Dickey 1984]. Ajoute des retards de
    $\Delta x_t$ au second membre pour absorber une autocorrélation de
    type ARMA des $u_t$ (résumé lu : un ARMA est approché par un
    autorégressif). Python : `statsmodels.tsa.stattools.adfuller` ; R :
    `tseries::adf.test`.
-   **KPSS** [KPSS92]. Hypothèses *inversées* : $H_0$ = stationnarité
    (autour d'un niveau ou d'une tendance), $H_1$ = racine unitaire.
    Statistique : somme des carrés des sommes partielles des résidus,
    normalisée par une variance de long terme (forme de mémoire, non
    vérifiée dans le texte). Python : `statsmodels.tsa.stattools.kpss` ;
    R : `urca::ur.kpss`.

**Lecture croisée.**

| ADF (H0 : racine unitaire) | KPSS (H0 : stationnaire) | Lecture |
|------------------------|------------------------|------------------------|
| rejette | ne rejette pas | stationnaire |
| ne rejette pas | rejette | racine unitaire |
| ne rejette pas | ne rejette pas | peu d'information (puissance faible) |
| rejette | rejette | modèle mal spécifié (rupture, tendance mal modélisée) |

**Limites.** Les tests usuels de racine unitaire ont peu de puissance
[KPSS92, introduction lue] ; en présence d'une rupture déterministe, ils
ne rejettent plus la racine unitaire même asymptotiquement [Per89]
(résumé lu). Le notebook (E3-B) montre les contradictions dans la zone
grise. **Ce ne sont pas des méthodes qui « pallient » l'AR** : ce sont
des *diagnostics* qui disent quelle classe de modèle choisir. La
réparation de l'inférence sous résidus autocorrélés est le rôle de
Newey–West (P11).

<a id="P11"></a> \## P11. Newey–West (HAC)

**Problème.** Pour $y=X\beta+u$ avec $u$ autocorrélé,
$\operatorname{Var}\hat\beta=(X^\top X)^{-1}\,X^\top\Omega X\,(X^\top X)^{-1}$,
où $\Omega$ est la covariance des erreurs ; la formule « iid » suppose
$\Omega=\sigma^2I$. *HAC* (*heteroskedasticity and autocorrelation
consistent*) désigne un estimateur de la variance de long terme des
scores $x_tu_t$, qui est ce que contient $X^\top\Omega X$.

**Estimateur** [NW87].
$\hat\Omega=\hat\Gamma_0+\sum_{j=1}^{L}\big(1-\tfrac j{L+1}\big)(\hat\Gamma_j+\hat\Gamma_j^\top)$,
avec $\hat\Gamma_j=\sum_tx_tu_t\,u_{t-j}x_{t-j}^\top$. Les poids de
Bartlett $1-j/(L+1)$ rendent l'estimateur semi-défini positif (résumé de
[NW87] lu : semi-défini positif et convergent ; les poids eux-mêmes :
texte non relu). Dans le notebook, $L=\lfloor\sqrt n\rfloor$ (réglage
arbitraire), appliqué aux scores $u_t\hat e_t$ de la pente.

**Ce qu'il fait et ne fait pas.** Il corrige l'*écart-type* de $\hat b$,
pas $\hat b$ lui-même, et ne modélise pas la dépendance : il n'améliore
pas les prévisions. Il est asymptotique ($L\to\infty$, $L/n\to0$) :
quand la longueur de corrélation $1/(1-\varphi)$ dépasse $L$, il reste
libéral. [KV05] (résumé lu) rapportent que les distorsions de taille,
selon la fenêtre, croissent avec la corrélation sérielle. Observé en
E3-A ($n=500$, $L=22$) : à $\varphi=0.98$ ($1/(1-\varphi)=50>L$), il
rejette 46 % au lieu de 5 %.

<a id="P12"></a> \## P12. Intervalles conformes : séparé et glissant

**Conforme séparé** (*split conformal* [LGRTW18], résumé lu : couverture
marginale en échantillon fini, pour tout estimateur de régression).
Algorithme : 1. Ajuster le modèle sur un bloc d'apprentissage ; le
figer. 2. Sur un bloc de calibration de $n$ points, calculer les
*scores* $V_i=|y_i-\hat y_i|$. 3. Poser $q=V_{(k)}$,
$k=\lceil(n+1)(1-\alpha)\rceil$ (statistique d'ordre). 4. Intervalle
pour une nouvelle cible : $\hat y\pm q$.

**Exemple.** $n=9$, scores triés
$0.1,\,0.2,\,0.3,\,0.4,\,0.5,\,0.6,\,0.8,\,1.1,\,2.0$, $\alpha=0.2$ :
$k=\lceil10\times0.8\rceil=8$, $q=1.1$.

**Garantie et preuve.** Si les $n+1$ scores (calibration et nouveau
point) sont échangeables et sans ex æquo, alors
$\mathbb P(V_{n+1}\le V_{(k)})=k/(n+1)\ge1-\alpha$. *Preuve* : par
échangeabilité, le rang de $V_{n+1}$ parmi les $n+1$ scores est uniforme
sur $\{1,\dots,n+1\}$ ; $V_{n+1}\le V_{(k)}$ équivaut à un rang $\le k$,
de probabilité $k/(n+1)$. ∎ La probabilité porte sur le tirage
*conjoint* de la calibration et du nouveau point : la garantie est
*marginale*, pas conditionnelle à un $x$ ni à un échantillon de
calibration donné. L'hypothèse d'échangeabilité n'apparaît pas dans les
résumés lus de [LGRTW18] et [RPC19] ; la preuve ci-dessus l'utilise.

**Conforme glissant.** Même recette, mais $q$ est recalculé à chaque
instant à partir des $W$ dernières erreurs *déjà observées* (au temps
$k$, la cible $y_{k-1}$ est connue). Les scores d'une série temporelle
ne sont pas échangeables avec le futur : la garantie est perdue, mais
l'intervalle suit les changements lents d'échelle (E2b : couverture 0.87
après une rupture, contre 0.60 pour le conforme séparé). Variante
adaptative [GC21] (résumé lu : en ligne, enveloppe de tout prédicteur,
couverture de long terme malgré les changements de loi) ; la règle de
mise à jour du niveau n'a pas été relue (non vérifié).

<a id="P13"></a> \## P13. Calibration, avec un exemple

Une prévision probabiliste est *calibrée* si les probabilités annoncées
se réalisent. **Exemple** : un prévisionniste qui annonce « 30 % de
pluie » est calibré si, sur tous les jours où il l'annonce, il pleut un
jour sur trois. Pour un intervalle à 90 % : la valeur observée doit
tomber dedans dans 90 % des cas. [GBR07] (résumé lu) distinguent la
*calibration* (accord statistique entre prévisions et observations) de
la *finesse* (étroitesse, propriété des seules prévisions) et proposent
de maximiser la finesse sous la contrainte de calibration. Version
probabiliste : si $F$ est la vraie fonction de répartition prédictive,
$F(Y)$ suit la loi uniforme (transformée intégrale de probabilité, PIT).

**Marginale contre conditionnelle.** La calibration *marginale* est une
moyenne sur tous les cas ; la *conditionnelle* est la même fréquence
dans chaque groupe de cas (ici : chaque niveau de volatilité). **Exemple
chiffré (E2).** L'intervalle gaussien de demi-largeur $c$ à variance
constante couvre $2\Phi(c/\sigma_t)-1$ quand la volatilité vaut
$\sigma_t$ : trop large quand elle est faible, trop étroit quand elle
est forte. Observé : 0.964, 0.921, 0.823 par tercile de volatilité, pour
0.903 en moyenne. Il est calibré marginalement (0.90) et non
conditionnellement. La perte pinball mélange calibration et finesse :
elle ne se lit pas seule.

<a id="P14"></a> \## P14. Protocole sans fuite : comment les données
servent

Pour **une** trajectoire simulée de
$N=N_{\rm tr}+N_{\rm cal}+N_{\rm te}$ points (E2 : 6000 + 1500 + 1500) :

```         
t = 1 ............ 6000 | 6001 ...... 7500 | 7501 ...... 9000
     apprentissage       calibration        test
     (ajustement)        (figé)             (évaluation)
```

1.  **Apprentissage.** On estime *une seule fois* tous les paramètres :
    moindres carrés de la moyenne, coefficient AR, paramètres GARCH. Ils
    sont ensuite figés.
2.  **Calibration** (E2 seulement). Le modèle figé prévoit à un pas
    chaque point du bloc de calibration ; les erreurs absolues servent à
    fixer la demi-largeur des intervalles conformes (P12). Les
    coefficients ne bougent pas ; seuls les retards AR lisent les
    observations les plus récentes.
3.  **Test.** Prévisions et métriques sur le dernier bloc. La prévision
    à l'origine $t$ n'utilise que $X_s$, $s\le t$. Pour le conforme
    glissant, la fenêtre de scores avance avec les erreurs déjà
    observées.

**On ne « refait » pas ce découpage sur les mêmes données.** On répète
tout le pipeline sur $R$ trajectoires *indépendantes* (P2), chacune avec
son propre apprentissage, sa propre calibration et son propre test.
**Garde-fou.** Une cellule brouille les observations après une origine
$t$ et vérifie que les prévisions aux origines $\le t$ ne changent pas :
c'est un test mécanique que la fonction de prévision ne lit pas
l'avenir.

------------------------------------------------------------------------

# Partie II — Preuves longues

<a id="T1"></a> \## T1. Causal et inversible

**Cadre.** $\varepsilon$ bruit blanc $(0,\sigma^2)$ ;
$\varphi(z)=1-\varphi_1z-\dots-\varphi_pz^p$ et
$\theta(z)=1+\theta_1z+\dots+\theta_qz^q$ sans racine commune ;
$\varphi(B)X_t=\theta(B)\varepsilon_t$.

**Proposition 1.** (i) Si $\varphi(z)\neq0$ pour $|z|\le1$, la série
$\psi(z)=\theta(z)/\varphi(z)=\sum_{j\ge0}\psi_jz^j$ vérifie
$\sum|\psi_j|<\infty$ et $X_t=\sum_{j\ge0}\psi_j\varepsilon_{t-j}$ est
l'unique solution stationnaire ; elle est causale. (ii) Si
$\theta(z)\ne0$ pour $|z|\le1$,
$\pi(z)=\varphi(z)/\theta(z)=\sum\pi_jz^j$ vérifie $\sum|\pi_j|<\infty$
et $\varepsilon_t=\sum_{j\ge0}\pi_jX_{t-j}$. Les réciproques sont
classiques ([BD91], Thm 3.1.1 et 3.1.2 ; non relu).

*Preuve de (i), sens direct.* $1/\varphi$ est analytique sur le disque
ouvert de rayon $\rho_0=\min_i|z_i|>1$ ($z_i$ : racines de $\varphi$),
donc son rayon de convergence dépasse 1 : ses coefficients $\chi_j$
décroissent géométriquement et $\sum|\chi_j|<\infty$. Le produit de
$\theta$ par $\chi$ est un produit de séries absolument convergentes :
$\sum|\psi_j|<\infty$ et
$\varphi(B)\psi(B)\varepsilon=\theta(B)\varepsilon$, de sorte que
$X=\psi(B)\varepsilon$ est solution. Unicité : si $Y$ est une autre
solution stationnaire, $\varphi(B)(X-Y)=0$ ; en appliquant $\chi(B)$,
$X-Y=\chi(B)\varphi(B)(X-Y)=0$ dans $L^2$. (ii) est identique en
échangeant $\varphi$ et $\theta$. ∎

**Le cas AR(1), dans les deux sens.**
$X_t=\varphi X_{t-1}+\varepsilon_t$. - $|\varphi|<1$ : en itérant,
$X_t=\sum_{j<m}\varphi^j\varepsilon_{t-j}+\varphi^mX_{t-m}$ ; comme
$\operatorname{Var}X$ est finie et $\varphi^m\to0$, le reste tend vers 0
dans $L^2$ : $X_t=\sum_{j\ge0}\varphi^j\varepsilon_{t-j}$ (causal). -
$|\varphi|>1$ : $X_t=\varphi^{-1}X_{t+1}-\varphi^{-1}\varepsilon_{t+1}$
; en itérant vers l'avenir,
$X_t=-\sum_{j\ge1}\varphi^{-j}\varepsilon_{t+j}$ est l'*unique* solution
stationnaire, et elle dépend de l'avenir du bruit : il n'y en a pas de
causale. - $|\varphi|=1$ : pas de solution stationnaire (marche
aléatoire).

**Proposition 2 (non-identifiabilité du second ordre).** Soit le MA(1)
$X_t=\varepsilon_t+\theta\varepsilon_{t-1}$ :
$\gamma(0)=\sigma^2(1+\theta^2)$, $\gamma(1)=\sigma^2\theta$. Le couple
$(\theta,\sigma^2)$ et le couple $(1/\theta,\ \sigma^2\theta^2)$ donnent
les mêmes $\gamma(0)=\sigma^2\theta^2(1+\theta^{-2})$ et
$\gamma(1)=\sigma^2\theta^2\theta^{-1}$. Exemple : $(0.5,\,1)$ et
$(2,\,0.25)$ donnent $(\gamma(0),\gamma(1))=(1.25,\,0.5)$. Une loi
gaussienne ne dépendant que de $\gamma$, la vraisemblance gaussienne a
deux modes. Heuristique :
$|1+\theta e^{-i\lambda}|=|\theta|\,|1+\theta^{-1}e^{-i\lambda}|$ ;
réfléchir une racine par rapport au cercle unité ne change que la phase
de la fonction de transfert, que le second ordre ne voit pas.

**Conséquences.** 1. *Convention.* On retient la représentation
inversible ($|\theta|<1$), car le bruit y est l'innovation. 2. *Variance
de l'innovation.* Pour $|\theta|>1$, la représentation inversible
équivalente est $(1/\theta,\ \sigma^2\theta^2)$ : l'erreur de prévision
à un pas a pour variance $\sigma^2\theta^2>\sigma^2$. Utiliser
$\sigma^2$ avec la représentation non inversible sous-estimerait la
variance d'erreur : la formule $\sigma^2\sum_{j<h}\psi_j^2$ (Idée 2)
n'est valable que pour la représentation à la fois causale et
inversible, dite *fondamentale*. 3. *Stabilité de la reconstruction du
bruit.* Pour un MA(1),
$\hat\varepsilon_t=X_t-\theta\hat\varepsilon_{t-1}$ ; une erreur
d'initialisation $e_0$ devient $(-\theta)^te_0$ : elle disparaît ssi
$|\theta|<1$. L'inversibilité est la stabilité de ce filtre, qui est
lui-même un réseau récurrent linéaire à une unité (§11.2 du document
compagnon).

**Théorème de Wold** (énoncé ; non vérifié ici). Tout processus
stationnaire au second ordre, purement non déterministe, s'écrit
$X_t=\sum_{j\ge0}\psi_j\varepsilon_{t-j}$ avec $\psi_0=1$,
$\sum\psi_j^2<\infty$, où
$\varepsilon_t=X_t-\mathbb E[X_t\mid X_{s},\ s<t]$ (projection linéaire)
est l'innovation, de variance égale à l'erreur quadratique de prévision
à un pas.

<a id="T2"></a> \## T2. Mémoire courte (ARMA) et mémoire longue (ARFIMA)

**ARMA : décroissance géométrique.** $\psi(z)=\theta(z)/\varphi(z)$ est
rationnelle, à pôles $z_i$ avec $|z_i|>1$ (causalité). Par décomposition
en éléments simples, pour $j$ grand, $\psi_j=\sum_ic_i(j)\,z_i^{-j}$ où
$c_i$ est un polynôme en $j$ de degré inférieur à la multiplicité du
pôle. Posons $r=\max_i|z_i|^{-1}<1$ et fixons $r'\in(r,1)$ :
$|\psi_j|\le Cr'^j$. Alors
$$|\gamma(h)|\le\sigma^2\sum_j|\psi_j||\psi_{j+h}|\le\sigma^2C^2r'^h\sum_jr'^{2j}=C'r'^h.$$
Donc **oui, c'est exponentiel** : $\rho(h)=O(r'^h)$, et
$\sum_h|\gamma(h)|<\infty$ (mémoire courte). Exemple de E1 :
$\rho(h)=\varphi^{h-1}\rho(1)$, soit
$\rho(24)\approx0.85^{23}\times0.902\approx0.02$.

**ARFIMA**$(0,d,0)$ : décroissance hyperbolique. Avec
$(1-B)^{-d}=\sum_j\psi_jB^j$, la série binomiale généralisée donne
$\psi_j=\dfrac{\Gamma(j+d)}{\Gamma(d)\,\Gamma(j+1)}\sim\dfrac{j^{d-1}}{\Gamma(d)}$.
Pour $0<d<\tfrac12$,
$$\gamma(h)=\sigma^2\sum_j\psi_j\psi_{j+h}\approx\frac{\sigma^2}{\Gamma(d)^2}\sum_jj^{d-1}(j+h)^{d-1}\sim\frac{\sigma^2}{\Gamma(d)^2}\,h^{2d-1}\int_0^\infty x^{d-1}(1+x)^{d-1}dx=\frac{\sigma^2B(d,1-2d)}{\Gamma(d)^2}\,h^{2d-1},$$
(en posant $j=hx$ ; l'intégrale est une intégrale d'Euler de première
espèce, convergente ssi $0<d<\tfrac12$). D'où $\rho(h)\sim c\,h^{2d-1}$
et, comme $2d-1>-1$, $\sum_h\rho(h)=\infty$ : mémoire longue. Cohérence
avec la forme exacte de Hosking,
$\rho(h)\sim\frac{\Gamma(1-d)}{\Gamma(d)}h^{2d-1}$ (de mémoire, non
comparée à [Hos81]) : on retrouve ce facteur avec
$\gamma(0)=\sigma^2\Gamma(1-2d)/\Gamma(1-d)^2$.

**Contraste chiffré.** À $h=100$ : AR(1) avec $\varphi=0.9$ :
$\rho=0.9^{100}\approx3\times10^{-5}$ ; ARFIMA avec $d=0.3$ :
$\rho\approx0.434\times100^{-0.4}\approx0.07$. La mémoire d'un ARFIMA se
mesure en centaines de pas, celle d'un ARMA en dizaines.

<a id="T3"></a> \## T3. GARCH(1,1) : représentation ARMA, moments
d'ordre 2 et 4

**Cadre.** $\varepsilon_t=\sigma_tz_t$,
$\sigma_t^2=\omega+a\varepsilon_{t-1}^2+b\sigma_{t-1}^2$, $\omega>0$,
$a,b\ge0$ ; $z_t$ i.i.d., $\mathbb Ez=0$, $\mathbb Ez^2=1$,
$\kappa=\mathbb Ez^4<\infty$ ($\kappa=3$ si gaussien) ;
$\mathcal F_t=\sigma(z_s,s\le t)$.

**(i) Structure.** $\sigma_t^2$ est $\mathcal F_{t-1}$-mesurable et
$z_t$ est indépendant de $\mathcal F_{t-1}$ :
$\mathbb E[\varepsilon_t\mid\mathcal F_{t-1}]=0$ et
$\mathbb E[\varepsilon_t^2\mid\mathcal F_{t-1}]=\sigma_t^2$. Comme
$\varepsilon_{t-1}^2=z_{t-1}^2\sigma_{t-1}^2$, on a
$\sigma_t^2=\omega+A_t\sigma_{t-1}^2$ avec $A_t=az_{t-1}^2+b$,
indépendant de $\sigma_{t-1}^2$ ; $\mathbb EA=a+b$ et
$\mathbb EA^2=\kappa a^2+2ab+b^2$.

**(ii) Moment d'ordre 2.** $m=\mathbb E\sigma^2$ vérifie
$m=\omega+(a+b)m$, donc $m=\omega/(1-a-b)$ si $a+b<1$ ; si $a+b\ge1$ il
n'existe pas de solution stationnaire de variance finie (car
$\omega>0$).

**(iii) Moment d'ordre 4.**
$M=\mathbb E\sigma^4=\omega^2+2\omega(a+b)m+\mathbb E[A^2]\,M$. Tous les
termes sont positifs : $M$ est fini ssi $\mathbb EA^2<1$, c'est-à-dire
$$\kappa a^2+2ab+b^2<1\qquad(\text{gaussien : }3a^2+2ab+b^2<1),$$ et
alors $M=\dfrac{\omega^2(1+a+b)}{(1-a-b)(1-\mathbb EA^2)}$. Comme
$\mathbb E\varepsilon^4=\kappa M$ et $(\mathbb E\varepsilon^2)^2=m^2$,
$$\text{kurtosis}(\varepsilon)=\kappa\,\frac{1-(a+b)^2}{1-\kappa a^2-2ab-b^2}\ \overset{\kappa=3}{=}\ 3\,\frac{1-(a+b)^2}{1-3a^2-2ab-b^2}.$$
Le kurtosis dépasse $\kappa$ ssi $a>0$ :
$1-\kappa a^2-2ab-b^2<1-(a+b)^2\iff(\kappa-1)a^2>0$. Même avec une loi
conditionnelle gaussienne, la loi marginale a des queues plus épaisses.
(Vérifié par le notebook pour $(a,b)=(0.10,0.85)$ : 3.774 théorique,
3.812 observé ; et par simulation hors notebook pour $(0.05,0.90)$ :
3.162 contre 3.162.)

**(iv) Représentation ARMA du carré.** Soit
$v_t=\varepsilon_t^2-\sigma_t^2=\sigma_t^2(z_t^2-1)$ :
$\mathbb E[v_t\mid\mathcal F_{t-1}]=0$ (différence de martingale), donc
les $v_t$ sont non corrélés, de variance $(\kappa-1)M<\infty$ sous (iii)
: un bruit blanc (non i.i.d.). En écrivant
$\sigma_{t-1}^2=\varepsilon_{t-1}^2-v_{t-1}$ :
$$\varepsilon_t^2=\sigma_t^2+v_t=\omega+a\varepsilon_{t-1}^2+b(\varepsilon_{t-1}^2-v_{t-1})+v_t=\omega+(a+b)\varepsilon_{t-1}^2+v_t-bv_{t-1}.$$
Donc $\varepsilon^2$ est un ARMA(1,1) de racine autorégressive $a+b$ et
de coefficient moyenne mobile $-b$.

**(v) Autocorrélation de** $\varepsilon^2$. Les formules du §3.1 du
document compagnon ($\varphi\to a+b$, $\theta\to-b$, donc
$\varphi+\theta=a$) donnent
$$\rho_{\varepsilon^2}(1)=\frac{a\,\big(1-b(a+b)\big)}{1-2b(a+b)+b^2},\qquad\rho_{\varepsilon^2}(h)=(a+b)^{h-1}\rho_{\varepsilon^2}(1).$$
Vérifié par simulation (20 000 chaînes) : $(a,b)=(0.05,0.90)$ : 0.0725
théorique, 0.0725 observé ; $(0.10,0.85)$ : 0.1791 contre 0.1779 ;
rapport $\rho(5)/\rho(1)$ : $(a+b)^4=0.815$ théorique, 0.809 à 0.813
observé. Les hypothèses de ces calculs sont celles de la représentation
ARMA : moment d'ordre 4 fini. Je n'ai pas comparé ces formules au texte
de [Bol86] (seul le résumé a été lu : conditions de stationnarité et
structure d'autocorrélation).

<a id="T4"></a> \## T4. Variance de la pente sous dépendance : la
variance de long terme

**Cadre.** $S$ stationnaire, $\sum_h|\gamma(h)|<\infty$,
$\omega^2:=\sum_{h\in\mathbb Z}\gamma(h)=2\pi f_S(0)$ ($f_S$ : densité
spectrale, avec la convention
$f_S(\lambda)=\frac1{2\pi}\sum_h\gamma(h)e^{-ih\lambda}$). Régresseur
centré $u_t=t/n-(n+1)/(2n)$, $\sum_tu_t=0$,
$\sum_tu_t^2=\frac{n^2-1}{12n}\sim\frac n{12}$. Sous $b=0$,
$\hat b=\sum_tu_tS_t/\sum_tu_t^2$ (le centrage rend l’intercepte sans
effet).

**Proposition.** $\operatorname{Var}(\hat b)\sum_tu_t^2\to\omega^2$.

*Preuve.*
$\operatorname{Var}(\hat b)=\sum_{s,t}u_su_t\gamma(s-t)/(\sum_tu_t^2)^2$
et $\sum_{s,t}u_su_t\gamma(s-t)=\sum_{|h|<n}\gamma(h)\,c_n(h)$ avec
$c_n(h)=\sum_tu_tu_{t+h}$ (somme sur les $t$ tels que $t,t+h\in[1,n]$).
(a) Par Cauchy–Schwarz, $|c_n(h)|\le\sum_tu_t^2$ pour tout $h$. (b) Pour
$h$ fixé, $u_{t+h}-u_t=h/n$, donc
$\big|c_n(h)-\sum_tu_t^2\big|\le(h/n)\sum_t|u_t|+h\max u^2\le h/2+h/4$,
négligeable devant $\sum_tu_t^2\sim n/12\to\infty$ :
$c_n(h)/\sum_tu_t^2\to1$. (c) Comme $\sum_h|\gamma(h)|<\infty$, le
théorème de convergence dominée (dominateur $|\gamma(h)|$, constante
$1$) donne
$\sum_h\gamma(h)\,c_n(h)/\sum_tu_t^2\to\sum_h\gamma(h)=\omega^2$. ∎

Il n'y a donc pas besoin de théorème spécifique sur les régressions à
tendance (type Grenander–Rosenblatt) : la convergence dominée suffit.

**Inflation pour un AR(1).** $\gamma(h)=\gamma(0)\varphi^{|h|}$ donne
$\omega^2=\gamma(0)\sum_h\varphi^{|h|}=\gamma(0)\frac{1+\varphi}{1-\varphi}$,
soit $r=\omega^2/\gamma(0)=(1+\varphi)/(1-\varphi)$ : 3, 19, 99 pour
$\varphi=0.5,\,0.9,\,0.98$ ; $r\to\infty$ quand $\varphi\to1$.
L'écart-type « iid » estime $\gamma(0)/\sum u^2$ au lieu de
$\omega^2/\sum u^2$ : la statistique $t$ a une variance $\approx r$ (sa
normalité asymptotique, par un théorème central limite pour sommes
pondérées de processus linéaires, n'est pas démontrée ici), et le test
nominal à 5 % rejette avec la probabilité $2\{1-\Phi(1.96/\sqrt r)\}$.
Taille d'échantillon efficace : $n/r$.

<a id="T5"></a> \## T5. Régression fallacieuse : divergence en $\sqrt n$
de la statistique $t$

**Cadre.** $S_t=\sum_{j\le t}\varepsilon_j$, $\varepsilon_j$ i.i.d.
centrés de variance $\sigma^2>0$. Régression de $S_t$ sur $(1,u_t)$ avec
$u_t$ comme en T4 ; $\hat b=\sum u_tS_t/\sum u_t^2$ (pente par unité de
$u=t/n$), résidus $\hat e_t=S_t-\bar S-\hat bu_t$,
$\hat s^2=(n-2)^{-1}\sum\hat e_t^2$,
$t_n=\hat b\big/\sqrt{\hat s^2/\sum u_t^2}$, et
$\mathrm{DW}_n=\sum_{t\ge2}(\hat e_t-\hat e_{t-1})^2/\sum\hat e_t^2$.

**Théorème.** Soit $W$ un mouvement brownien standard,
$I=\int_0^1(r-\tfrac12)W(r)\,dr$ et $V=\int_0^1(W-\bar W)^2dr-12I^2$, où
$\bar W=\int_0^1W$. Alors $V>0$ presque sûrement et, quand $n\to\infty$
:

1.  $n^{-1/2}\hat b\Rightarrow12\sigma I\sim N\big(0,\tfrac65\sigma^2\big)$
    ;
2.  $n^{-1}\hat s^2\Rightarrow\sigma^2V$ ;
3.  $n^{-1/2}t_n\Rightarrow\sqrt{12}\,I/\sqrt V$, variable aléatoire
    sans atome en 0 : $|t_n|\to\infty$ en probabilité à la vitesse
    $\sqrt n$ et $\mathbb P(|t_n|>c)\to1$ pour tout $c$ ;
4.  $n\cdot\mathrm{DW}_n\Rightarrow1/V$ : la statistique de
    Durbin–Watson tend vers 0 comme $1/n$.

*Preuve.*

**Étape 1 (principe d'invariance de Donsker, non vérifié ici).**
$X_n(r)=S_{\lfloor nr\rfloor}/(\sigma\sqrt n)\Rightarrow W$ dans
$D[0,1]$. Pour $g$ lipschitzienne,
$\frac1n\sum_tg(t/n)X_n(t/n)=\int_0^1g(r)X_n(r)\,dr+O_p(1/n)$ (somme de
Riemann d'une fonction en escalier). Toutes les quantités ci-dessous
sont des fonctionnelles continues de $X_n$ pour la norme uniforme, et
$W$ est continu : on peut passer à la limite par le théorème de
l'application continue.

**Étape 2 (pente).** $u_t=(t/n-\tfrac12)-\tfrac1{2n}$, donc
$\sum_tu_tS_t=\sigma\sqrt n\sum_t(t/n-\tfrac12)X_n(t/n)+O_p(\sqrt n)=\sigma n^{3/2}\int_0^1(r-\tfrac12)X_n(r)dr+O_p(\sqrt n)$.
Avec $\sum_tu_t^2\sim n/12$ :
$\hat b=12\sigma\sqrt n\int_0^1(r-\tfrac12)X_n+o_p(\sqrt n)$, d'où 1. La
variance de $I$ :
$\iint(r-\tfrac12)(s-\tfrac12)\min(r,s)\,dr\,ds=\int_0^1(r-\tfrac12)\big(\tfrac{r^2}4-\tfrac{r^3}6\big)dr=\tfrac1{120}$
(calcul :
$\int_0^1(s-\tfrac12)\min(r,s)\,ds=\tfrac{r^2}{4}-\tfrac{r^3}{6}$), donc
$\operatorname{Var}(12\sigma I)=144\sigma^2/120=\tfrac65\sigma^2$.
(Vérifié par simulation : variance de $\hat b/\sqrt n$ entre 1.16 et
1.28 pour $n=100$ à $6400$, théorie 1.2 ; erreur-type d'environ 0.05
avec 1000 répliques.)

**Étape 3 (résidus).** $\hat e_t$ est le résidu de la projection de
$S_t$ sur $\mathrm{vect}(1,u_t)$ pour le produit scalaire de comptage ;
asymptotiquement c'est le résidu de la projection de $\sigma\sqrt n\,W$
sur $\mathrm{vect}(1,r-\tfrac12)$ dans $L^2[0,1]$. Les composantes sont
$\bar W$ et $12I$ ($\int(r-\tfrac12)^2=\tfrac1{12}$), d'où
$\sum_t\hat e_t^2/(\sigma^2n^2)\Rightarrow\int(W-\bar W)^2-12I^2=V$
(Pythagore), c'est-à-dire 2. $V>0$ p.s. car une trajectoire brownienne
n'est presque sûrement pas une fonction affine.

**Étape 4 (**$t$).
$t_n=\hat b\sqrt{\sum u_t^2}/\hat s\approx\dfrac{12\sigma\sqrt n\,I\,\sqrt{n/12}}{\sigma\sqrt{nV}}=\sqrt n\,\dfrac{\sqrt{12}\,I}{\sqrt V}$.
La loi de $I$ est gaussienne non dégénérée : $\mathbb P(I=0)=0$, donc la
limite de $|t_n|/\sqrt n$ n'a pas d'atome en 0 et, par le théorème de
Portmanteau,
$\mathbb P(|t_n|>c)=\mathbb P(|t_n|/\sqrt n>c/\sqrt n)\to\mathbb P(\sqrt{12}|I|/\sqrt V>0)=1$.

**Étape 5 (Durbin–Watson).**
$\hat e_t-\hat e_{t-1}=\varepsilon_t-\hat b/n$, donc
$\sum_{t\ge2}(\hat e_t-\hat e_{t-1})^2=\sum\varepsilon_t^2-2\tfrac{\hat b}n\sum\varepsilon_t+(n-1)\tfrac{\hat b^2}{n^2}=n\sigma^2(1+o_p(1))$,
car $\hat b/n=O_p(n^{-1/2})$ rend les deux derniers termes $O_p(1)$.
Avec $\sum\hat e_t^2\sim\sigma^2n^2V$, on obtient 4. ∎

**Contrôle numérique (cellule E3-D, 1000 marches par** $n$).

| $n$ | $\mathbb P(\lvert t\rvert>1.96)$ | médiane de $\lvert t\rvert/\sqrt n$ | médiane de $n\cdot\mathrm{DW}$ | $\operatorname{Var}(\hat b/\sqrt n)$ (théorie 1.2) |
|---------------|---------------|---------------|---------------|---------------|
| 100 | 0.871 | 0.81 | 17.9 | 1.16 |
| 400 | 0.935 | 0.86 | 18.6 | 1.24 |
| 1600 | 0.974 | 0.93 | 17.6 | 1.28 |
| 6400 | 0.990 | 0.88 | 18.7 | 1.24 |

Le taux de rejet tend vers 1 ; $\lvert t\rvert/\sqrt n$ et
$n\cdot\mathrm{DW}$ se stabilisent (erreurs-types : 0.011 sur les taux,
environ 0.05 sur la variance). Cohérence entre 2 et 4 :
$n\cdot\mathrm{DW}\approx1/V$ et
$\hat s^2/n\approx\sigma^2V\approx0.056$ en médiane, soit
$1/V\approx18$.

**Remarques.** (a) Pour un processus linéaire d'intégration 1 de
variance de long terme $\omega^2$, on attend le même résultat avec
$\sigma$ remplacé par $\omega$ (non démontré ici). (b) Newey–West ne
règle pas le problème : sa statistique ne diverge plus mais a une loi
limite non standard (non démontrée) ; observé à $n=500$ : 0.767 de
rejets à 5 %. (c) Après différenciation, $\Delta S_t=\varepsilon_t$ est
i.i.d. et la régression sur une constante n'est plus fallacieuse.
