Méthodes usuelles pour les séries temporelles : tour d'horizon sur données synthétiques

> Document compagnon du notebook [`notebooks/00b_synthetic_methods.ipynb`](../notebooks/00b_synthetic_methods.ipynb) du projet `01-ts-uncertainty` (paquet `tsunc`). Version du 2026-11-18. Les définitions préalables et les preuves longues sont dans l'[annexe technique](00b_annexe_technique.md) : les liens « [P3] », « [T5] »… y renvoient.
>
> **Statut des références (§9).** ✔ texte lu : le passage utilisé a été lu dans le texte. ✔ résumé lu : seul le résumé a été lu. ◐ fiche seule : notice bibliographique seule. ◐ ouvrage non relu. *non vérifié* : cité de mémoire, sans source relue. Les formules sont démontrées ici ou en annexe, puis contrôlées par simulation (tableau « théorie contre simulation » du notebook). Un résultat classique dont la source n'a pas été relue est signalé « non vérifié ».

## 0. En bref

**Question.** Sur des données dont la vérité est connue, qu'apporte chaque couche de modélisation usuelle (moyenne, dépendance linéaire, variance conditionnelle, régimes, tendance), et où cesse-t-elle de tenir sa promesse ? Les séries sont synthétiques et génériques : une série horaire à cycle quotidien (\$m=24\$). Le lien avec la climatologie est un horizon d'application (Blagnac), pas une contrainte du cadre. Aucune décision en aval n'est évaluée.

**Trois mots, avant les résultats.** Un intervalle de prédiction *couvre* quand la valeur observée tombe dedans ; la *couverture*est la fréquence de ces cas (cible : \$1-\\alpha\$, ici 0.90). Un intervalle *conforme* est de la forme « prévision ± quantile empirique des erreurs passées ». Il est *séparé* (*split*) quand ce quantile est calculé une fois sur un bloc de calibration réservé, *glissant* (*rolling*) quand il est recalculé à chaque instant sur les \$W\$ dernières erreurs déjà observées ([P12](00b_annexe_technique.md#P12)).

**Résultats** (moyennes de \$R=20\$ répliques sauf mention ; chiffres détaillés au §8).

1.  *Mémoire courte (E1, ARMA(1,1)).* La persistance bat la climatologie à \$h=1,2\$ (MASE 0.37 et 0.61 contre 0.72), puis la perd dès \$h=3\$. « Climatologie + AR(3) » suit l'oracle à moins de 0.01 de MASE (\$h=1\$ : 0.296 contre 0.296 ; \$h=24\$ : 0.718 contre 0.714). Toutes les courbes rejoignent la climatologie (théorie : 0.715) quand la mémoire est épuisée.

2.  *Variance conditionnelle (E2, AR-GARCH).* L'intervalle gaussien à variance constante et le conforme séparé couvrent 0.90 en moyenne, mais 0.96 quand la volatilité est faible et 0.82 quand elle est forte. GARCH et oracle couvrent 0.90 partout. Les largeurs moyennes (3.27 et 3.30 contre 3.20) ne montrent presque rien ; le score d'intervalle (4.27 contre 4.00) le montre.

3.  *Rupture de variance (E2b).* Écart-type de l'innovation ×2 (variance ×4) juste après la calibration : couverture ≈ 0.60 pour les méthodes figées, 0.87 pour le conforme glissant (0.91 après 250 h), 0.83 pour GARCH, rappelé vers l'ancien niveau de variance (point fixe 3 au lieu de 4).

4.  *Régimes (E2c).* Une chaîne de Markov à deux régimes donne une loi prédictive bimodale. À \$h=24\$, l'intervalle gaussien AR à 90 % couvre 0.953 des observations (trop large), mais celui à 50 % n'en couvre que 0.316 (trop étroit) : une gaussienne représente mal cette loi, car ses quantiles centraux tombent dans le creux entre les deux modes (§3.3 le démontre).

5.  *Tendance ou aléa (E3).* Avec un bruit AR(1) (\$\\varphi=0.5/0.9/0.98\$), le test \$t\$ « iid » de la pente rejette « pas de tendance » dans 24/65/85 % des cas alors que la pente est nulle (théorie : 26/65/84 %) ; Newey–West ramène à 8/18/46 %. ADF et KPSS, aux hypothèses nulles opposées, se contredisent près de la racine unitaire. Pour une marche aléatoire, la statistique \$t\$ diverge en \$\\sqrt n\$ ([T5](00b_annexe_technique.md#T5), vérifié en E3-D). Enfin, à \$h=100\$, on regarde la fréquence avec laquelle la valeur observée à \$t+100\$ tombe dans l'intervalle à 90 % : la recette « tendance + AR(1) » appliquée à une marche aléatoire couvre 0.50 ; la recette « marche aléatoire » appliquée à une tendance stationnaire couvre 1.00, avec des intervalles de largeur moyenne près de 8 fois plus grande (35.7 contre 4.6).

6.  *Lien avec les réseaux de neurones (§11).* Un réseau est une famille paramétrique entraînée avec une perte ; les résultats 1 à 4 fixent le plancher (oracle) à battre, et l'AR(1) est un réseau récurrent linéaire dont le gradient suit la fonction d'autocorrélation.

Lecture courte : §5 (largeur moyenne, calibration, score d'intervalle), §6 (tendance ou aléa), §11 (réseaux).

## 1. Cadre et hypothèses

### 1.1 Hypothèses

-   **(H1) Information.** Temps discret, espace \$(\\Omega,\\mathcal F,\\mathbb P)\$ ; \$\\mathcal F_t=\\sigma(X_s,\\ s\\le t)\$ est l'information du prévisionniste à l'origine \$t\$. Un *oracle* peut disposer d'une information plus grande \$\\mathcal G_t\\supseteq\\mathcal F_t\$ (états cachés).

-   **(H2) Décomposition.** \$X_t=\\mu_t+S_t\$, avec \$\\mu_t=\\beta\^\\top d(t)\$ déterministe et \$d(t)=\\big(1,\\ t/n,\\ \\cos\\tfrac{2\\pi kt}{m},\\ \\sin\\tfrac{2\\pi kt}{m}\\big)\_{k=1,2}\$, où \$n\$ est la longueur de la série et \$m=24\$. Valeurs : \$\\beta=(10,,2,,2.1,,2.1,,0.7,,0)\$ (niveau, pente totale sur l'échantillon, deux harmoniques journalières).

-   **(H3) Aléa.** \$S\$ est centré et *stationnaire au second ordre* : \$\\mathbb E S_t\^2\<\\infty\$, et \$\\operatorname{Cov}(S_t,S\_{t+h})=\\gamma(h)\$ ne dépend que du décalage \$h\$, pas de \$t\$ (oui : « la covariance ne dépend que du lag », avec une moyenne constante et une variance finie ; [P1](00b_annexe_technique.md#P1)). Exceptions : E2b (rupture de variance), E3 (racine unitaire, rupture de niveau).

-   **(H4) Protocole sans fuite.** Pour **une** trajectoire simulée, découpée en trois blocs consécutifs (E2 : 6000, 1500, 1500 points) :

    ```         
    apprentissage (6000) | calibration (1500) | test (1500)
    ```

    1.  *Apprentissage* : on estime **une seule fois** les coefficients de \$\\mu\$ (moindres carrés), les coefficients AR et les paramètres GARCH, puis on les fige.

    2.  *Calibration* : le modèle figé prévoit à un pas chaque point du bloc ; les erreurs servent à fixer la largeur des intervalles conformes. Aucun coefficient n'est réajusté : le bloc d'apprentissage ne « prédit » pas le bloc suivant par un nouvel entraînement.

    3.  *Test* : prévisions et métriques sur le dernier bloc. La prévision à l'origine \$t\$ n'utilise que \$X_s\$, \$s\\le t\$.

    On ne refait pas ce découpage sur les mêmes données : on répète tout le pipeline sur \$R\$ trajectoires indépendantes ([P2](00b_annexe_technique.md#P2), [P14](00b_annexe_technique.md#P14)). Un garde-fou automatisé brouille les données après une origine et vérifie que les prévisions aux origines antérieures ne bougent pas.

### 1.2 Ce qu'on estime : la loi prédictive

La cible est la loi prédictive \$P\_{t,h}=\\mathcal L(X\_{t+h}\\mid\\mathcal F_t)\$. Toute prévision (point, quantile, intervalle) est une fonctionnelle de \$P\_{t,h}\$, c'est-à-dire une application qui associe un nombre ou un intervalle à une loi : l'espérance, un quantile. L'*oracle* est cette fonctionnelle calculée avec les vrais paramètres et, selon le cas, les vrais états cachés ; il donne le plancher du risque pour toute perte propre.

> **Idée 1 — la perte détermine la fonctionnelle estimée.** *Quadratique.* Pour \$Y\\in L\^2\$ et une sous-tribu \$\\mathcal G\$, \$\\mathbb E[Y\\mid\\mathcal G]\$ est la projection orthogonale de \$Y\$ sur \$L\^2(\\mathcal G)\$. Pour \$f\\in L\^2(\\mathcal G)\$, Pythagore donne \$\\mathbb E(Y-f)\^2=\\mathbb E(Y-m)\^2+\\mathbb E(m-f)\^2\$ avec \$m=\\mathbb E[Y\\mid\\mathcal G]\$ : le minimiseur est \$m\$, et le risque excédentaire de \$f\$ est \$\\mathbb E(m-f)\^2\$. (Sous bruit gaussien, c'est aussi le maximum de vraisemblance.) *Pinball.* Pour \$\\tau\\in(0,1)\$, \$\\rho\_\\tau(u)=u,(\\tau-\\mathbf 1\_{u\<0})\$. Si \$F\$ (fonction de répartition conditionnelle) est continue, \$\\partial_q,\\mathbb E,\\rho\_\\tau(Y-q)=-\\tau,\\mathbb P(Y\>q)+(1-\\tau),\\mathbb P(Y\<q)=F(q)-\\tau\$, qui s'annule au quantile d'ordre \$\\tau\$ et change de signe : c'est un minimum. *Propriété.* Ces pertes sont *propres* (leur espérance est minimale quand la prévision est la vraie loi) [GR07 ; la pinball est la perte de la régression quantile, [KB78]]. Avec une perte non propre, une prévision fausse peut gagner. La vraisemblance logarithmique est propre par l'inégalité de Gibbs (\$\\mathbb E\\log(q/p)\\le\\log\\mathbb E(q/p)=0\$, Jensen).*Repère historique* (non vérifié, de mémoire) : les moindres carrés remontent à Legendre et Gauss, vers 1805–1809 ; le minimiseur de la pinball est l'objet de la régression quantile de [KB78] (fiche lue).

### 1.3 Risque, moyenne empirique, et pourquoi \$R\$ répétitions

Une *procédure* est une recette complète : estimer les paramètres sur un bloc d'apprentissage \$D\$, puis prévoir. Deux espérances sont à distinguer ([P2](00b_annexe_technique.md#P2)) :

-   le risque du prédicteur ajusté, \$R(\\hat f_D)=\\mathbb E\\big[\\ell(X\_{t+h},\\hat f_D(X\_{\\le t}))\\mid D\\big]\$ : le prédicteur est fixé, on moyenne sur l'origine de test et le futur ;

-   le risque de la procédure, \$\\bar R=\\mathbb E_D,R(\\hat f_D)\$ : on moyenne en plus sur l'aléa d'apprentissage ; c'est lui qui compare les méthodes.

**Exemple.** AR(1), prédicteur \$\\hat\\varphi X_t\$, perte quadratique : \$R(\\hat f_D)=\\sigma\^2+\\gamma(0)(\\hat\\varphi-\\varphi)\^2\$, qui dépend de \$D\$ par \$\\hat\\varphi\$. Comme \$\\mathbb E(\\hat\\varphi-\\varphi)\^2\\approx(1-\\varphi\^2)/n_0\$ (pour \$n_0\$ points d'apprentissage ; approximation classique, non démontrée ici) et \$\\gamma(0)(1-\\varphi\^2)=\\sigma\^2\$, on a \$\\bar R\\approx\\sigma\^2(1+1/n_0)\$ : estimer le paramètre coûte une fraction \$1/n_0\$ du risque irréductible.

**Ce que fait l'ergodicité.** La moyenne empirique \$\\hat R_n=n\^{-1}\\sum\_{i\\le n}\\ell_i\$ sur \$n\$ origines d'une trajectoire de test converge vers \$R(\\hat f_D)\$ pour un prédicteur *figé* et un processus stationnaire ergodique (théorème ergodique : la moyenne temporelle tend vers l'espérance). Une trajectoire suffit donc pour le risque *de ce prédicteur*, pas pour celui de la procédure. D'où les \$R\$ répétitions :

1.  chaque trajectoire tire son propre \$\\hat\\varphi\$ : on moyenne sur l'aléa d'apprentissage ;

2.  les erreurs d'une trajectoire sont dépendantes, et l'erreur-type naïve \$\\hat s/\\sqrt n\$ est trop petite d'un facteur de l'ordre de \$\\sqrt r\$, \$r\$ étant la variance de long terme relative (Idée 8) ; \$R\$ trajectoires indépendantes donnent une erreur-type valide \$\\hat s/\\sqrt R\$ sans modéliser la dépendance.

Le notebook rapporte la moyenne des \$\\hat R_n\$ sur \$R\$ trajectoires avec cette erreur-type ; un écart inférieur à deux erreurs-types n'est pas interprétable.

## 2. Dénombrement des familles de dépendance

Un générateur AR(1) ne couvre qu'une famille de dépendance. Voici les familles usuelles ; la dernière colonne dit ce que le notebook en fait. Les familles les moins connues sont décrites dans l'annexe (liens dans la colonne 3).

| Famille | Modèle ou mécanisme | Ce qu'elle capture, pourquoi elle compte | Ici |
|----|----|----|----|
| ARMA, ARIMA, SARIMA | \$\\phi(B)S_t=\\theta(B)\\varepsilon_t\$ (\$B\$ : opérateur retard) ; \$(1-B)\^d(1-B\^m)\^D\$ | Mémoire linéaire à décroissance *géométrique*(\$\\rho(h)=O(r\^h)\$, \$r\<1\$, [T2](00b_annexe_technique.md#T2)) ; racines unitaires (tendance et saisonnalité stochastiques) ; condition de causalité et d'inversibilité ([P3](00b_annexe_technique.md#P3), [T1](00b_annexe_technique.md#T1)) [BJRL15 ; BD91] | E1 (ARMA(1,1)), E3 (marche aléatoire) |
| ARFIMA | \$(1-B)\^dS_t=\\varepsilon_t\$, \$0\<d\<\\tfrac12\$ | Mémoire longue : \$\\rho(h)\\sim c,h\^{2d-1}\$ (décroissance *hyperbolique*), autocovariances non sommables ; la décroissance en loi de puissance est démontrée en [T2](00b_annexe_technique.md#T2), la forme exacte de la constante est à relire dans [Hos81] ; [GJ80] | non traité |
| Composantes inobservées, espace d'état | \$X_t=\\mu_t+\\varepsilon_t\$, \$\\mu_t=\\mu\_{t-1}+\\eta_t\$ (niveau local) ; filtre de Kalman [DK12] | Rend \$\\mu_t\$ *aléatoire* ; la tendance déterministe est le cas \$\\sigma\_\\eta\^2=0\$ ; cadre naturel de « \$\\mu\$ ou aléa ? » ([P5](00b_annexe_technique.md#P5)) | E3 |
| ARCH, GARCH | \$\\varepsilon_t=\\sigma_tz_t\$, \$\\sigma_t\^2=\\omega+a,\\varepsilon\_{t-1}\^2+b,\\sigma\_{t-1}\^2\$ | Variance conditionnelle qui se regroupe ; queues épaisses [Eng82 ; Bol86] | E2, E2b |
| Volatilité stochastique | \$X_t=e\^{h_t/2}z_t\$, \$h_t\$ : AR(1) latent | Variance non prévisible exactement ; vraisemblance sans forme close (filtrage) [KSC98] ; contraste avec GARCH en [P6](00b_annexe_technique.md#P6) | non traité |
| Régimes de Markov, générateurs de temps | chaîne cachée \$s_t\$, matrice \$P\$ ; loi de \$X_t\$ sachant \$s_t\$ | Lois prédictives en *mélange*, multimodales ; état discret (pluie ou non) [Ham89 ; Ric81 ; WW99] ; vocabulaire en [P4](00b_annexe_technique.md#P4) | E2c |
| Modèles à seuil (TAR, STAR) | régime fixé par une variable *observée*, p. ex. \$X\_{t-d}\>r\$ ; \$r\$ est estimé | Non-linéarité à régimes observables [TL80] ; origine du seuil en [P7](00b_annexe_technique.md#P7) | non traité |
| Processus gaussiens | \$S\\sim\\mathrm{GP}(0,k)\$ : la covariance est le noyau \$k(s,t)\$ | Dépendance par noyau ; OU = noyau de Matérn \$\\nu=\\tfrac12\$ [RW06] ; [P8](00b_annexe_technique.md#P8) | OU (vérification) |
| EDS | \$dX=a(X),dt+b(X),dW\$ : OU, CIR, double puits | Temps continu, échantillonnage arbitraire, lois stationnaires explicites ; lien avec la diffusion en apprentissage profond [KP92 ; UO30 ; BSB05 ; SSKKEP21] ; [P9](00b_annexe_technique.md#P9) | générateur et vérification |
| Dynamique déterministe non linéaire | \$x\_{t+1}=f(x_t)\$ chaotique + bruit d'observation | Aléa apparent ; prévisibilité bornée par l'exposant de Lyapunov | non traité |
| Processus ponctuels auto-excités | intensité \$\\lambda_t=\\lambda_0+\\sum\_{t_i\<t}g(t-t_i)\$ (Hawkes) | Grappes d'événements | non traité |
| Multivarié | VAR, facteurs dynamiques, copules | Dépendance entre séries | non traité |
| Non-stationnarités déterministes | ruptures, amplitude saisonnière variable | Fausse toute inférence qui suppose la stationnarité ; sous une rupture de tendance, les tests de racine unitaire ne rejettent plus la racine unitaire, même asymptotiquement [Per89] (résumé lu) | E2b, E3 |

Les lignes « dynamique déterministe non linéaire », « processus ponctuels auto-excités » et « multivarié » sont citées sans source. Outils équivalents. Python : `statsmodels` (`ARIMA`, `UnobservedComponents`, `MarkovRegression`), `arch` (GARCH), `sklearn.gaussian_process` (noyau `Matern(nu=0.5)`). R : `forecast::Arima`, `rugarch::ugarchfit`, `quantreg::rq`, `tseries::adf.test`, `urca::ur.kpss`, `sandwich::NeweyWest`. Seuls `scikit-learn`, `statsmodels` et `scipy` sont utilisés ici ; `scikit-learn` n'a ni ARMA ni GARCH.

**Mémoire courte, mémoire longue.** Pour un ARMA, \$\\rho(h)\$ décroît *géométriquement*, comme \$r\^h\$ avec \$r\<1\$ ; pour un ARFIMA, \$\\rho(h)\\sim c,h\^{2d-1}\$ décroît comme une puissance, et \$\\sum_h\\rho(h)\$ diverge. À \$h=100\$ : \$0.9\^{100}\\approx3\\times10\^{-5}\$ pour un AR(1) de coefficient 0.9, contre environ 0.07 pour un ARFIMA avec \$d=0.3\$ ([T2](00b_annexe_technique.md#T2)).

Pourquoi ces familles-là dans le notebook : chacune met en défaut un composant d'une méthode usuelle. ARMA : l'horizon et la mémoire. GARCH : la largeur constante des intervalles. Régimes : la gaussianité. EDS : l'échantillonnage et la discrétisation. Marche aléatoire et tendance : l'hypothèse « \$\\mu\$ est déterministe ».

## 3. Quatre familles en détail

### 3.1 ARMA(1,1)

**Hypothèses.** \$\\varepsilon_t\$ bruit blanc gaussien \$N(0,\\sigma\^2)\$ ; \$\|\\varphi\|\<1\$ (*causalité* : \$S_t\$ ne dépend que du passé du bruit) ; \$\|\\theta\|\<1\$ (*inversibilité* : le bruit se reconstruit à partir des observations passées, donc \$\\varepsilon_t\$ est \$\\mathcal F_t\$-mesurable) ; \$\\varphi+\\theta\\neq0\$. Ces deux conditions et ce qui arrive sans elles sont démontrés en [T1](00b_annexe_technique.md#T1).

\$\$S_t=\\varphi S\_{t-1}+\\varepsilon_t+\\theta\\varepsilon\_{t-1}.\$\$

Écriture de Wold : \$S_t=\\sum\_{j\\ge0}\\psi_j\\varepsilon\_{t-j}\$ avec \$\\psi_0=1\$ et \$\\psi_j=\\varphi\^{j-1}(\\varphi+\\theta)\$ pour \$j\\ge1\$. On en tire \$\$\\gamma(0)=\\sigma\^2\\sum_j\\psi_j\^2=\\sigma\^2,\\frac{1+2\\varphi\\theta+\\theta\^2}{1-\\varphi\^2},\\qquad \\rho(1)=\\frac{(1+\\varphi\\theta)(\\varphi+\\theta)}{1+2\\varphi\\theta+\\theta\^2},\\qquad \\rho(h)=\\varphi\^{h-1}\\rho(1)\\ \\ (h\\ge1).\$\$

**Prévision.** \$\\mathbb E[S\_{t+h}\\mid\\mathcal F_t]=\\varphi\^{h-1}(\\varphi S_t+\\theta\\varepsilon_t)\$ et \$\\operatorname{Var}\\big(S\_{t+h}-\\mathbb E[S\_{t+h}\\mid\\mathcal F_t]\\big)=\\sigma\^2\\sum\_{j\<h}\\psi_j\^2\$, qui croît vers \$\\gamma(0)\$.

> **Idée 2 — les poids de Wold sont la mémoire.** Tout processus stationnaire purement non déterministe est un filtre causal de ses innovations, c'est-à-dire de ses erreurs de prévision à un pas (Wold, 1938 ; non vérifié).*Preuve de la prévision.* Dans \$S\_{t+h}=\\sum\_{j\\ge0}\\psi_j\\varepsilon\_{t+h-j}\$, les termes \$j\<h\$ portent sur des innovations futures, indépendantes de \$\\mathcal F_t\$ et centrées ; les termes \$j\\ge h\$ sont \$\\mathcal F_t\$-mesurables, car \$\\varepsilon_s\$ est \$\\mathcal F_s\$-mesurable par inversibilité. Donc \$\\mathbb E[S\_{t+h}\\mid\\mathcal F_t]=\\sum\_{j\\ge h}\\psi_j\\varepsilon\_{t+h-j}\$ et l'erreur est \$\\sum\_{j\<h}\\psi_j\\varepsilon\_{t+h-j}\$, de variance \$\\sigma\^2\\sum\_{j\<h}\\psi_j\^2\$ (innovations non corrélées). Pour l'ARMA(1,1), \$\\sum\_{j\\ge h}\\psi_j\\varepsilon\_{t+h-j}=\\varphi\^{h-1}(\\varphi+\\theta)\\sum\_{i\\ge0}\\varphi\^i\\varepsilon\_{t-i}=\\varphi\^{h-1}(\\varphi S_t+\\theta\\varepsilon_t)\$. À l'horizon où \$\\sum\_{j\<h}\\psi_j\^2\\approx\\sum_j\\psi_j\^2\$, le passé n'apporte plus rien : on retombe sur la climatologie. *Repère historique* (non vérifié, de mémoire) : Yule (1927) ajuste un AR(2) au cycle des taches solaires, premier modèle autorégressif ; Slutsky (1937) montre qu'une somme mobile de bruit blanc produit des oscillations apparentes. Les deux répondent à la même question : comment des fluctuations quasi périodiques naissent d'un bruit.

**L'oracle ne fait pas d'inversion de matrice.** Pour un processus gaussien général, la prévision exacte est \$\\Sigma\_{t+h,1:t}\\Sigma\_{1:t}\^{-1}S\_{1:t}\$ (une matrice \$t\\times t\$ à inverser, coût \$O(t\^3)\$). Pour un ARMA, la récurrence \$\\hat\\varepsilon_s=S_s-\\varphi S\_{s-1}-\\theta\\hat\\varepsilon\_{s-1}\$ donne le même résultat en \$O(t)\$, grâce à l'inversibilité ; et dans E1, l'oracle lit directement l'innovation vraie du générateur ([P3](00b_annexe_technique.md#P3)).

**Persistance contre climatologie.** Erreur de la persistance : \$X\_{t+h}-X_t=(\\mu\_{t+h}-\\mu_t)+(S\_{t+h}-S_t)\$, d'erreur quadratique \$\\Delta\\mu_h\^2+2\\gamma(0)(1-\\rho(h))\$, où \$\\Delta\\mu_h\^2\$ est la moyenne sur les origines de \$(\\mu\_{t+h}-\\mu_t)\^2\$. Erreur de la climatologie : \$\\gamma(0)\$. La persistance gagne si et seulement si \$\$\\rho(h)\>\\tfrac12+\\frac{\\Delta\\mu_h\^2}{2\\gamma(0)}.\$\$ Avec \$\\varphi=0.85\$, \$\\theta=0.30\$, la condition est vraie à \$h=1,2\$ et fausse dès \$h=3\$ ; le MASE observé le confirme (§8).

**MASE de la climatologie.** Le MASE (erreur absolue moyenne rapportée à celle de la naïve saisonnière sur l'apprentissage ; [HK06], §3.4 de [HA18]) a pour échelle \$\\mathbb E\|S\_{t+m}-S_t\|\$ à la pente près. Pour un bruit gaussien, \$\\mathrm{MASE}*{\\rm clim}=\\sqrt{\\gamma(0)}/\\sqrt{2\\gamma(0)(1-\\rho(m))}=1/\\sqrt{2(1-\\rho(24))}=0.715\$ ici. Le MASE de l'oracle à l'horizon \$h\$ vaut \$\\sigma\\sqrt{\\sum*{j\<h}\\psi_j\^2}/\\sqrt{2\\gamma(0)(1-\\rho(m))}\$.

**Vérifié par simulation** (200 000 points) : \$\\operatorname{Var}(S)\$ 5.766 contre 5.780 ; \$\\rho(1)\$ 0.902 contre 0.902 ; \$\\rho(6)\$ 0.400 contre 0.397.

### 3.2 GARCH(1,1)

**Hypothèses.** \$z_t\$ i.i.d., centré, de variance 1 (gaussien ici) ; \$\\omega\>0\$, \$a\\ge0\$, \$b\\ge0\$. Les coefficients sont notés \$(\\alpha,\\beta)\$ dans [Bol86] et dans le code ; on écrit \$(a,b)\$ ici pour réserver \$\\alpha\$ au niveau d'erreur des intervalles.

\$\$\\varepsilon_t=\\sigma_tz_t,\\qquad \\sigma_t\^2=\\omega+a,\\varepsilon\_{t-1}\^2+b,\\sigma\_{t-1}\^2.\$\$

\$\\sigma_t\^2\$ est \$\\mathcal F\_{t-1}\$-mesurable, \$\\varepsilon_t\$ est une différence de martingale et \$\\operatorname{Var}(\\varepsilon_t\\mid\\mathcal F\_{t-1})=\\sigma_t\^2\$. Stationnarité au second ordre si et seulement si \$a+b\<1\$, avec \$\\bar\\sigma\^2=\\mathbb E,\\varepsilon\^2=\\omega/(1-a-b)\$. Le moment d'ordre 4 est fini si \$3a\^2+2ab+b\^2\<1\$, et le kurtosis vaut alors \$3,\\dfrac{1-(a+b)\^2}{1-3a\^2-2ab-b\^2}\>3\$ : les queues sont épaisses bien que la loi conditionnelle soit gaussienne (preuve : [T3](00b_annexe_technique.md#T3)).

**Prévision de la variance.** \$\\mathbb E[\\sigma\^2\_{t+h}\\mid\\mathcal F_t]=\\bar\\sigma\^2+(a+b)\^{h-1}(\\sigma\^2\_{t+1}-\\bar\\sigma\^2)\$. À \$h=1\$, la loi prédictive est exactement \$N(\\cdot,\\sigma\_{t+1}\^2)\$ ; à \$h\\ge2\$, c'est un mélange d'échelles de gaussiennes (non évalué ici).

> **Idée 3 — non corrélé ne veut pas dire indépendant.** Posons \$v_t=\\varepsilon_t\^2-\\sigma_t\^2=\\sigma_t\^2(z_t\^2-1)\$, différence de martingale. En remplaçant \$\\sigma\_{t-1}\^2=\\varepsilon\_{t-1}\^2-v\_{t-1}\$ dans \$\\varepsilon_t\^2=\\sigma_t\^2+v_t\$ : \$\$\\varepsilon_t\^2=\\omega+(a+b),\\varepsilon\_{t-1}\^2+v_t-b,v\_{t-1}.\$\$ Le carré des innovations suit donc un ARMA(1,1) de racine autorégressive \$a+b\$ [Bol86] (la preuve complète, avec les moments d'ordre 4, est en [T3](00b_annexe_technique.md#T3)). La dépendance est dans le second moment, pas dans le premier : les \$\\varepsilon_t\$ sont non corrélés mais non indépendants. Conséquence pour les intervalles : la couverture *marginale* est la moyenne de couvertures *conditionnelles* qui varient avec \$\\sigma_t\$. Un intervalle de largeur constante \$2c\$ couvre \$2\\Phi(c/\\sigma_t)-1\$ quand la volatilité vaut \$\\sigma_t\$ : trop large quand elle est faible, trop étroit quand elle est forte, avec la bonne moyenne. *Repère historique.* [Eng82] introduit l'ARCH pour modéliser la variance de l'inflation britannique (résumé lu) ; [Bol86] généralise en GARCH. Le problème résolu : la variance d'erreur d'une prévision n'est pas constante, elle dépend de l'histoire récente.

**Estimation.** Quasi-vraisemblance gaussienne \$-\\tfrac12\\sum_t(\\log\\sigma_t\^2+\\varepsilon_t\^2/\\sigma_t\^2)\$, maximisée par `scipy.optimize` ; la récurrence est vectorisée par `scipy.signal.lfilter`. La cohérence de cet estimateur sans hypothèse gaussienne sur \$z_t\$ est un résultat classique que je n'ai pas vérifié.

**Dans le notebook.** \$S_t=0.7,S\_{t-1}+\\varepsilon_t\$, \$(\\omega,a,b)=(0.05,,0.10,,0.85)\$ : \$\\bar\\sigma\^2=1\$, \$a+b=0.95\$, kurtosis théorique 3.774. L'oracle connaît les paramètres : \$\\sigma\_{t+1}\^2\$ est alors \$\\mathcal F_t\$-mesurable (les \$\\varepsilon_s=S_s-0.7S\_{s-1}\$ se déduisent des observations).

**Vérifié par simulation** : \$\\mathbb E,\\varepsilon\^2\$ 1.000 contre 0.998 ; kurtosis 3.774 contre 3.812 ; \$\\mathbb E[\\sigma\^2\_{t+24}\\mid\\sigma\^2\_{t+1}=3]\$ 1.615 contre 1.611.

### 3.3 Régimes de Markov et générateurs de temps

**Hypothèses.** Chaîne \$(s_t)\$ à \$K\$ états, de matrice de transition \$P\_{ik}=\\mathbb P(s\_{t+1}=k\\mid s_t=i)\$, avec trois propriétés ([P4](00b_annexe_technique.md#P4)) : *homogène* (\$P\$ ne dépend pas de \$t\$), *irréductible* (de tout état on atteint tout état avec une probabilité positive), *apériodique* (les durées de retour à un état ont pour pgcd 1). Elle admet alors une loi stationnaire unique \$\\pi\$ (\$\\pi P=\\pi\$). « Sachant la chaîne » veut dire : conditionnellement à toute la trajectoire des états latents (non observés), les \$X_t\$ sont indépendants, \$X_t\\mid s_t=k\\sim N(\\mu_k,\\sigma_k\^2)\$. Marginalement, ils sont dépendants : toute la dépendance vient de la chaîne cachée.

Pour \$P\$ diagonalisable, \$P\^h\\to\\mathbf 1\\pi\$ à la vitesse \$\|\\lambda_2\|\^h\$, où \$\\lambda_2\$ est la deuxième valeur propre en module. La loi prédictive sachant l'état courant \$i\$ est le mélange fini \$\$\\mathcal L(X\_{t+h}\\mid s_t=i)=\\sum\_{k}(P\^h)*{ik},N(\\mu_k,\\sigma_k\^2).\$\$ Si l'état n'est pas observé, les poids \$(P\^h)*{ik}\$ sont remplacés par \$\\sum_i\\mathbb P(s_t=i\\mid\\mathcal F_t)(P\^h)\_{ik}\$, où les probabilités filtrées viennent du filtre de Hamilton [Ham89] (non implémenté ici).

**Cas du notebook.** Deux états, probabilité de rester 0.95, \$\\mu=(-2,2)\$, \$\\sigma_k=1\$ : \$\\pi=(\\tfrac12,\\tfrac12)\$, \$\\lambda_2=2\\cdot0.95-1=0.9\$, \$\\operatorname{Var}(X)=1+4=5\$ et \$\\rho_X(h)=4\\lambda_2\^h/5=0.8\\cdot0.9\^h\$ pour \$h\\ge1\$.

> **Idée 4 — l'autocorrélation est aveugle à la forme de la loi.** Cette fonction d'autocorrélation est celle d'un AR(1) de coefficient 0.9 observé avec un bruit blanc (un ARMA(1,1) gaussien). Ce processus gaussien a une loi prédictive unimodale ; la chaîne de Markov a une loi bimodale. Deux processus de même structure du second ordre ont donc des quantiles prédictifs différents. *Calcul qui explique E2c.* À \$h=24\$, \$\\rho_X(24)=0.064\$ : la loi prédictive est presque la loi stationnaire, le mélange \$\\tfrac12N(-2,1)+\\tfrac12N(2,1)\$ de variance 5. Un intervalle gaussien centré de variance 5 est \$\\pm1.508\$ à 50 % (largeur 3.016) et \$\\pm3.678\$ à 90 % (largeur 7.356). Sa couverture sous le mélange est \$\\tfrac12\\sum\_{m=\\pm2}[\\Phi(c-m)-\\Phi(-c-m)]\$ : 0.311 à 50 %, 0.953 à 90 %. Le notebook observe 0.316 et 0.953, avec des largeurs 3.006 et 7.331. L'intervalle à 50 % tombe dans le creux entre les deux modes (en \$\\pm2\$) ; celui à 90 % est trop large parce qu'il doit atteindre les deux modes. Un intervalle calibré sur la variance seule a la bonne largeur à un niveau et une mauvaise couverture à l'autre.

**Générateurs de temps.** Ils combinent une chaîne de Markov pour l'état discret (occurrence de pluie) et un modèle conditionnel à cet état pour les variables continues [Ric81 ; revue : WW99]. Je n'ai pas relu ces articles : ce paragraphe décrit le principe général, pas leur contenu. [Ham89] utilise des régimes de Markov pour le cycle économique (résumé lu).

**Vérifié par simulation** : \$\\mathbb P(s=1)\$ 0.500 contre 0.499 ; \$\\rho_X(5)\$ 0.472 contre 0.475 ; \$\\rho_X(24)\$ 0.064 contre 0.064.

### 3.4 Équations différentielles stochastiques

**Hypothèses.** \$W\$ mouvement brownien, coefficients lipschitziens (existence et unicité d'une solution forte) ; observations sur une grille de pas \$\\Delta\$. Notation et exemples (OU, CIR, double puits) : [P9](00b_annexe_technique.md#P9).

**Ornstein–Uhlenbeck.** \$dS=-\\kappa S,dt+\\sigma,dW\$, \$\\kappa\>0\$. La solution exacte (formule d'Itô appliquée à \$e\^{\\kappa t}S_t\$) donne la transition gaussienne \$\$S\_{t+\\Delta}\\mid S_t\\sim N!\\Big(e\^{-\\kappa\\Delta}S_t,\\ \\ \\sigma\^2,\\frac{1-e\^{-2\\kappa\\Delta}}{2\\kappa}\\Big),\$\$ c'est-à-dire un AR(1) de coefficient \$\\varphi=e\^{-\\kappa\\Delta}\$. La loi stationnaire est \$N(0,\\sigma\^2/2\\kappa)\$ et la covariance \$\\frac{\\sigma\^2}{2\\kappa}e\^{-\\kappa\|\\tau\|}\$ (noyau de Matérn \$\\nu=\\tfrac12\$ [RW06]).

> **Idée 5 — un AR(1) est un OU échantillonné.** L'AR(1) du monde discret est exactement l'OU observé au pas \$\\Delta\$, avec \$\\varphi=e\^{-\\kappa\\Delta}\\in(0,1)\$ : un \$\\hat\\varphi\\le0\$ contredit le modèle. L'échelle de mémoire \$1/\\kappa\$ ne dépend pas du pas d'échantillonnage, contrairement à \$\\varphi\$ : c'est elle qu'on compare entre jeux de données. Par invariance du maximum de vraisemblance, \$\\hat\\kappa=-\\log\\hat\\varphi/\\Delta\$.

**Euler–Maruyama.** \$S\_{k+1}=S_k-\\kappa S_k\\Delta+\\sigma\\sqrt\\Delta,\\xi_k\$ est un AR(1) de coefficient \$1-\\kappa\\Delta\$, de variance stationnaire \$\\sigma\^2/\\big(\\kappa(2-\\kappa\\Delta)\\big)\$. Le rapport à la variance exacte vaut \$1/(1-\\kappa\\Delta/2)\\approx1+\\kappa\\Delta/2\$ : le schéma gonfle la variance de \$\\kappa\\Delta/2\$ au premier ordre. Il est stable si \$\\kappa\\Delta\<2\$.

**Double puits.** \$dX=(X-X\^3),dt+\\sigma,dW=-V'(X),dt+\\sigma,dW\$ avec \$V(x)=\\tfrac{x\^4}4-\\tfrac{x\^2}2\$ (puits en \$\\pm1\$, barrière \$\\Delta V=\\tfrac14\$). La densité invariante annule le flux de l'équation de Fokker–Planck stationnaire, \$\\tfrac12\\sigma\^2p'+V'p=0\$, d'où \$p(x)\\propto\\exp(-2V(x)/\\sigma\^2)\$. La trajectoire saute entre les puits à une fréquence réglée par \$\\Delta V\$ comparée à \$\\sigma\^2/2\$ (heuristique). Échantillonné au pas \$\\Delta\$, c'est un AR(1) *non linéaire* de loi stationnaire bimodale : aucun ARMA n'en rend compte.

> **Idée 6 — pourquoi les EDS reviennent avec la diffusion.** Un processus de diffusion est un processus de Markov dont la densité de transition obéit aux équations de Kolmogorov (Fokker–Planck). Les modèles génératifs à base de score font évoluer les données vers du bruit par une EDS directe, puis les régénèrent par l'EDS rétrograde, qui exige le score \$\\nabla\\log p_t\$ [SSKKEP21]. Ce chapitre est donc le socle mathématique de la reproduction « diffusion » du plan d'architectures historiques (§11.4).

> *Repères historiques (non vérifiés, de mémoire).* Langevin (1908) écrit l'équation d'une particule brownienne soumise à un frottement et à une force aléatoire ; Uhlenbeck et Ornstein (1930) en étudient la loi [UO30, fiche partielle]. Vasicek (1977) utilise l'OU pour les taux d'intérêt ; Hasselmann (1976) lit le climat comme l'intégrale lente du bruit météorologique ; Kramers (1940) donne le taux de passage d'une barrière, proportionnel à \$e\^{-2\\Delta V/\\sigma\^2}\$ à l'ordre dominant ; Benzi, Sutera et Vulpiani (1982) proposent la résonance stochastique d'un double puits pour les cycles glaciaires. Dans tous les cas, le problème résolu par la modélisation est le même : un système lent ou à plusieurs équilibres, forcé par un bruit rapide.

**Température.** Le modèle « moyenne saisonnière + OU » est celui de [BSB05] : moyenne et volatilité saisonnières, avec un bruit de type OU piloté par un processus de Lévy hyperbolique généralisé (résumé lu). [CD05] étudient la dynamique de la moyenne et de la variance conditionnelles de la température quotidienne (résumé lu).

**Pourquoi aucune méthode de prévision n'est évaluée sur les EDS ?** Quatre raisons, de nature différente.

1.  *Périmètre convenu.* Lors de la validation du plan, les EDS ont été retenues en « générateur et vérification ». C'est une décision de portée, pas une impossibilité.

2.  *Pour l'OU, l'évaluation serait redondante.* La transition exacte est un AR(1) (Idée 5) : l'oracle à \$h\$ pas est \$N(\\varphi\^hS_t,\\ v(1-\\varphi\^{2h}))\$ avec \$v=\\sigma\^2/2\\kappa\$, et les méthodes de E1 s'y appliquent telles quelles.

3.  *Pour le double puits, l'oracle n'a pas de forme close.* La loi de transition à \$h\$ pas résout l'équation de Fokker–Planck ; pour juger des méthodes, il faut d'abord la calculer (résolution numérique sur grille, ou Monte Carlo : de nombreuses trajectoires issues de \$x_t\$). C'est faisable, mais ce n'est pas écrit.

4.  *Ce qu'on attend, sans l'avoir testé.* Par analogie avec E2c, à horizon grand devant le temps de passage entre puits la loi prédictive tend vers la loi invariante bimodale, qu'une gaussienne AR représente mal ; à horizon court elle est unimodale autour du puits courant. Un « E4 » (oracle Monte Carlo, gaussien AR, conforme, régression quantile) est le prolongement naturel.

**Vérifié par simulation** : variance OU exacte (\$\\Delta=0.5\$) 0.497 contre 0.500 ; Euler (\$\\Delta=0.5\$) 0.666 contre 0.667 ; Euler (\$\\Delta=0.1\$) 0.529 contre 0.526 ; distance en variation totale entre l'histogramme du double puits (\$\\sigma=0.6\$, 500 000 pas) et \$p\$ : 0.015 (bruit d'échantillonnage et biais du schéma ne sont pas séparés).

## 4. Les méthodes usuelles, et pourquoi elles comptent

Deux définitions rapides pour les lignes « conforme ». *Séparé* : on ajuste le modèle sur un bloc, on calcule les erreurs absolues sur un bloc de calibration, on prend leur quantile d'ordre \$\\lceil(n+1)(1-\\alpha)\\rceil/n\$, et l'intervalle est « prévision ± ce quantile » (exemple chiffré et preuve : [P12](00b_annexe_technique.md#P12)). *Glissant* : même recette, mais le quantile est recalculé à chaque instant sur les \$W\$ dernières erreurs déjà observées.

| Méthode | Ce qu'elle fait | Équivalents (sklearn, R) | Pourquoi elle compte | Première limite observée |
|----|----|----|----|----|
| Persistance | \$\\hat X\_{t+h\\mid t}=X_t\$ | `numpy` ; R `forecast::naive` | Référence gratuite ; difficile à battre à très court terme quand \$\\rho(1)\$ est grand | Erreur \$\\Delta\\mu_h+S\_{t+h}-S_t\$ : biais saisonnier et variance \$2\\gamma(0)(1-\\rho(h))\$ ; perdue dès \$h=3\$ (E1) |
| Naïve saisonnière | \$\\hat X\_{t+h\\mid t}=X\_{t+h-m\\lceil h/m\\rceil}\$ | R `forecast::snaive` ; échelle du MASE | Référence standard des séries saisonnières | Recopie une seule observation bruitée : erreur quadratique \$2\\gamma(0)(1-\\rho(m))\\approx2\\gamma(0)\$, double de la climatologie quand \$\\rho(24)\\approx0.02\$ (E1) |
| Climatologie harmonique | \$\\hat\\mu_t=\\hat\\beta\^\\top d(t)\$, moindres carrés sur l'apprentissage | `LinearRegression(fit_intercept=False)`; R `lm(x ~ t + cos + sin)` | Meilleur prédicteur à long horizon : tout le signal est dans \$\\mu\$ | Ignore l'anomalie courante ; extrapole la tendance telle quelle (E3) |
| Climatologie + AR direct | \$\\hat X\_{t+h}=\\hat\\mu\_{t+h}+\\hat\\gamma_h\^\\top(r_t,\\dots,r\_{t-p+1})\$, \$r=X-\\hat\\mu\$, une régression par horizon | `LinearRegression` sur matrice de retards ; R `lm` sur retards | Capte la mémoire courte ; la stratégie *directe*n'enchaîne pas les erreurs | \$p\$ fixé ; un modèle par \$h\$ ; ne donne pas de loi |
| Intervalle gaussien, variance constante | \$\\hat y\\pm z\\hat s\$, \$\\hat s\$ = écart-type des résidus d'apprentissage | `scipy.stats.norm.ppf` | Intervalle paramétrique standard (ARIMA) | Couverture conditionnelle non uniforme sous GARCH (E2) ; loi mal représentée sous régimes (E2c) |
| Conforme séparé | \$\\hat y\\pm\\hat q\$, \$\\hat q\$ = statistique d'ordre \$\\lceil(n+1)(1-\\alpha)\\rceil\$ des \$\\lvert\$résidus\$\\rvert\$ de calibration | trois lignes de `numpy` | Couverture *marginale*garantie sans hypothèse de loi, sous échangeabilité [LGRTW18] | L'échangeabilité est fausse en série temporelle ; marginal seulement ; ne suit pas une rupture (E2, E2b) |
| Conforme glissant | même rang, sur les \$W\$ derniers résidus | idem | S'adapte aux changements d'échelle lents | La garantie est perdue ; compromis sur \$W\$ (E2b : \$W=500\$) |
| Régression quantile linéaire | \$q\_\\tau(r\\mid v_t)=a\_\\tau+b\_\\tau v_t\$ ; \$v_t\$ = moyenne des \$\\lvert r\\rvert\$ sur 24 h | `QuantileRegressor(quantile=tau, alpha=0, solver="highs")` ; R `quantreg::rq` | Modélise directement les quantiles, avec la perte du futur réseau à sortie quantile [KB78] | Dépend du proxy ; linéaire ; croisement de quantiles possible |
| GARCH(1,1), quasi-vraisemblance | \$\\hat\\sigma_t\^2\$ récursive ; \$\\hat y\\pm z\\hat\\sigma_t\$ | `scipy.optimize` + `lfilter` (rien dans sklearn) ; Python `arch` ; R `rugarch` | Prévoit la volatilité elle-même | Gaussienne ; paramètres figés ; une rupture le ramène vers l'ancien niveau (E2b) |

**Sur les références conformes.** Les conformes sont des intervalles « prévision ponctuelle ± quantile empirique des erreurs » ; la garantie en échantillon fini pour tout estimateur de régression est celle de [LGRTW18] (résumé lu), prouvée ici sous échangeabilité en [P12](00b_annexe_technique.md#P12). [BCRT23] (résumé lu) étendent l'analyse aux données non échangeables par des quantiles pondérés et une randomisation ; que le conforme glissant en soit un cas particulier n'est pas vérifié. [GC21] adaptent le niveau en ligne sous changement de loi (ACI ; résumé lu, règle de mise à jour non relue). [RPC19] combinent conforme et régression quantile (CQR, méthode du projet).

## 5. Métriques

| Métrique | Définition | Mesure | Piège | Fonction |
|----|----|----|----|----|
| MASE |  \$\mathrm{MAE}*{*\mathrm{test}} / \mathrm{MAE}{\mathrm{train}}\^{\text{na\"{\i}ve saisonni\`ere } m}\$ [HK06 ; HA18] | Erreur ponctuelle sans unité ; \$\<1\$ : mieux que la naïve saisonnière | Dépend de l'échelle choisie ; ne dit rien de l'incertitude | `tsunc.metrics.mase` |
| Pinball | \$\\overline{\\rho\_\\tau(y-q\_\\tau)}\$, moyenne sur observations et niveaux | Qualité d'un quantile (perte propre) | Ne se lit pas seule : mélange calibration et finesse | `tsunc.metrics.pinball_loss` |
| Couverture | \$n\^{-1}\\sum_i\\mathbf 1{l_i\\le y_i\\le u_i}\$ | Fréquence d'inclusion ; cible \$1-\\alpha\$ | *Marginale* : une moyenne ; la *conditionnelle* (par groupe de volatilité) peut être très différente | `tsunc.metrics.coverage` |
| Largeur moyenne | \$n\^{-1}\\sum_i(u_i-l_i)\$ | Finesse (*sharpness*), dans l'unité de la série | Seule, ne dit rien : l'intervalle vide a la largeur 0 | `tsunc.metrics.mean_width` |
| Score d'intervalle | \$\\mathrm{IS}*\\alpha=(u-l)+\\tfrac2\\alpha(l-y)*++\\tfrac2\\alpha(y-u)\_+\$ [GR07] | Largeur et erreurs de couverture en un nombre ; propre ; à minimiser | Pondère les ratés par \$2/\\alpha=20\$ | `interval_score` (notebook) |

*Orientation du score.* Dans la version du rapport technique de [GR07] que j'ai lue, la règle est orientée en score à maximiser, \$S=-2\\alpha,\\mathrm{IS}*\\alpha\$ ; on utilise ici la forme \$\\mathrm{IS}*\\alpha\$ ci-dessus, à minimiser, qui est celle de la table. Le texte publié n'a pas été relu : non vérifié.

### 5.1 Que veut dire « largeur moyenne de l'intervalle » ?

Pour des intervalles \$[l_i,u_i]\$ sur \$n\$ cibles, c'est \$\\frac1n\\sum_i(u_i-l_i)\$, exprimée dans l'unité de la série. C'est ce que [GBR07] appellent la *finesse* (*sharpness*), une propriété des prévisions seules, sans référence aux observations. Leur principe : maximiser la finesse sous la contrainte de calibration. Étroit ne veut donc pas dire meilleur en soi.

-   Un intervalle de largeur 0 est parfaitement fin et ne couvre presque jamais.

-   Pour un oracle gaussien de variance constante, la largeur à 90 % est \$2z\\sigma=3.29,\\sigma\$ (\$z=1.645\$) : c'est un plancher, pas un défaut. En E2, la largeur moyenne de l'oracle vaut 3.205 \$=2z,\\mathbb E[\\sigma_t]\$ avec \$\\mathbb E[\\sigma_t]=0.974\<1\$ (inégalité de Jensen).

-   E2c, \$h=24\$, niveau 50 % : l'intervalle gaussien est plus étroit que l'oracle (3.01 contre 3.98) et ne couvre que 0.316 au lieu de 0.505.

-   E2 : 3.27 contre 3.20 de largeur moyenne séparent à peine l'intervalle gaussien de l'oracle, alors que sa couverture conditionnelle va de 0.96 à 0.82.

Conclusion : lire la largeur *avec* la couverture, conditionnelle si possible, ou par un score qui combine les deux.

### 5.2 Calibration, avec un exemple

Une prévision probabiliste est *calibrée* si les probabilités annoncées se réalisent. Exemple : un prévisionniste qui annonce « 30 % de pluie » est calibré si, sur tous les jours où il l'annonce, il pleut un jour sur trois. Pour un intervalle à 90 % : la valeur observée doit tomber dedans dans 90 % des cas. [GBR07] (résumé lu) distinguent la calibration (accord statistique entre prévisions et observations) de la finesse. Version probabiliste : si \$F\$ est la vraie fonction de répartition prédictive, \$F(Y)\$ suit la loi uniforme (transformée intégrale de probabilité, PIT) ; les histogrammes PIT sont le test usuel.

La calibration *marginale* est une moyenne sur tous les cas ; la *conditionnelle* est la même fréquence dans chaque groupe (ici : chaque tercile de volatilité). En E2, l'intervalle gaussien couvre 0.964, 0.921, 0.823 par tercile pour 0.903 en moyenne : calibré marginalement, non conditionnellement ([P13](00b_annexe_technique.md#P13)).

### 5.3 Score d'intervalle et pinball

> **Idée 7 — le score d'intervalle est la perte pinball des deux bornes.** Avec \$\\rho\_\\tau(u)=u(\\tau-\\mathbf 1\_{u\<0})\$, on a \$\\rho\_{\\alpha/2}(y-l)=\\tfrac\\alpha2(y-l)+(l-y)*+\$ et \$\\rho*{1-\\alpha/2}(y-u)=(y-u)*+-\\tfrac\\alpha2(y-u)\$. Leur somme vaut \$\\tfrac\\alpha2(u-l)+(l-y)*++(y-u)*+\$, donc \$\$\\mathrm{IS}*\\alpha(l,u;y)=\\tfrac2\\alpha\\big[\\rho\_{\\alpha/2}(y-l)+\\rho\_{1-\\alpha/2}(y-u)\\big],\\qquad \\overline{\\mathrm{IS}}=\\tfrac4\\alpha\\cdot\\overline{\\text{pinball}}\\ \\text{(moyenne sur les deux niveaux).}\$\$ La condition du premier ordre \$\\partial_u,\\mathbb E[\\mathrm{IS}]=1-\\tfrac2\\alpha,\\mathbb P(Y\>u)=0\$ donne \$\\mathbb P(Y\>u)=\\alpha/2\$ : le score est minimal pour les vrais quantiles. Le notebook vérifie l'identité numériquement et réutilise `tsunc.metrics.pinball_loss` ; seule l'enveloppe d'une ligne `interval_score` est nouvelle.

Autres métriques non utilisées : le CRPS, que [GR07] écrivent comme une intégrale de scores de Brier sur les seuils (texte lu) ; son expression comme intégrale de la pinball sur tous les niveaux, à un facteur 2 près, est une identité classique que je n'ai pas vérifiée dans le texte (non vérifié).

## 6. Tendance déterministe ou aléa ?

### 6.1 Deux modèles et une question

-   **(D)** \$X_t=a+b,t/n+S_t\$, \$S\$ stationnaire (tendance déterministe, aléa stationnaire).

-   **(R)** \$X_t=X\_{t-1}+b/n+\\varepsilon_t\$ (marche aléatoire avec dérive : la tendance est un cumul d'aléas).

Les deux produisent des trajectoires qui « montent ». La question *tendance déterministe ou stochastique ?* est posée pour les séries macroéconomiques par [NP82] (résumé lu) : la réponse commande la façon de prévoir et de quantifier l'incertitude. Aucun test n'a une puissance égale à 1 à \$n\$ fini : deux lois gaussiennes non dégénérées sur \$\\mathbb R\^n\$ sont mutuellement absolument continues. La séparation n'est qu'asymptotique (\$n\\to\\infty\$, paramètres fixes).

### 6.2 Variance de la pente sous dépendance

Soit \$u_t=t/n-\\bar u\$ le régresseur centré et \$\\hat b=\\sum_tu_tX_t/\\sum_tu_t\^2\$. Sous (D) avec \$b=0\$, \$\$\\operatorname{Var}(\\hat b)=\\frac{\\sum\_{s,t}u_su_t\\gamma(s-t)}{(\\sum_tu_t\^2)\^2}\\ \\sim\\ \\frac{\\sum\_{h\\in\\mathbb Z}\\gamma(h)}{\\sum_tu_t\^2}=\\frac{2\\pi f_S(0)}{\\sum_tu_t\^2}.\$\$ *Preuve en trois lignes* ([T4](00b_annexe_technique.md#T4) pour le détail). Le numérateur vaut \$\\sum\_{\|h\|\<n}\\gamma(h)c_n(h)\$ avec \$c_n(h)=\\sum_tu_tu\_{t+h}\$. Par Cauchy–Schwarz, \$\|c_n(h)\|\\le\\sum_tu_t\^2\$ ; pour \$h\$ fixé, \$u\$ varie de \$h/n\$ d'un pas à l'autre, donc \$c_n(h)/\\sum_tu_t\^2\\to1\$. Comme \$\\sum_h\|\\gamma(h)\|\<\\infty\$, la convergence dominée conclut. Ici \$2\\pi f_S(0)=\\sum_h\\gamma(h)\$ est la *variance de long terme*.

> **Idée 8 — la variance de long terme, pas la variance.** L'écart-type « iid » de `OLS` estime \$\\gamma(0)/\\sum u\^2\$, pas \$2\\pi f_S(0)/\\sum u\^2\$. Le rapport \$r=2\\pi f_S(0)/\\gamma(0)=\\sum_h\\rho(h)\$ vaut \$(1+\\varphi)/(1-\\varphi)\$ pour un AR(1) : 3 pour \$\\varphi=0.5\$, 19 pour 0.9, 99 pour 0.98. La statistique \$t\$ a une variance \$\\approx r\$ et non 1 : le test de niveau nominal 5 % rejette avec la probabilité \$2{1-\\Phi(1.96/\\sqrt r)}\$, soit 0.258, 0.653, 0.844. La taille d'échantillon efficace est \$n/r\$ : environ 5 observations indépendantes pour \$n=500\$ et \$\\varphi=0.98\$. Quand \$\\varphi\\to1\$, \$r\\to\\infty\$ : c'est la régression fallacieuse du §6.3.

**Newey–West** [NW87] est un estimateur *HAC* (*heteroskedasticity and autocorrelation consistent*) : il remplace \$\\gamma(0)\$ par une estimation de \$\\sum_h\\gamma(h)\$ avec des poids de Bartlett \$1-j/(L+1)\$ ([P11](00b_annexe_technique.md#P11)). Il pallie la dépendance des erreurs *dans l'inférence* (l'écart-type de \$\\hat b\$), sans la modéliser ni améliorer la prévision. Il est asymptotique : [KV05] (résumé lu) rapportent des distorsions de taille qui croissent avec la corrélation sérielle, et E3-A l'observe (\$n=500\$, \$L=22\$, \$\\varphi=0.98\$ : 46 % de rejets au lieu de 5 %, alors que la longueur de corrélation \$1/(1-\\varphi)=50\$ dépasse \$L\$).

### 6.3 Régression fallacieuse

**Énoncé.** Si \$S\$ est une marche aléatoire, \$\\hat b\$ (pente en unités de \$u=t/n\$) est de l'ordre de \$\\sqrt n\$ alors que son écart-type « iid » reste de l'ordre de 1 : la statistique \$t\$ diverge comme \$\\sqrt n\$, le taux de rejet de « pente nulle » tend vers 1, la statistique de Durbin–Watson tend vers 0 comme \$1/n\$. C'est la limite \$\\varphi\\to1\$ de l'Idée 8 : \$r\\to\\infty\$.

**Heuristique.** Une marche aléatoire n'a pas de moyenne vers laquelle revenir : sur de longues plages elle monte ou descend, et ressemble à une tendance. Un régresseur lisse (le temps) « explique » cette dérive, et les moindres carrés l'interprètent comme du signal, alors que l'aléa est cumulé : la variance de \$S_t\$ croît en \$t\\sigma\^2\$, pas la variance d'une série stationnaire autour d'une droite.

**Histoire.**

-   *Yule (1926)* décrit des « corrélations absurdes » entre séries temporelles (non vérifié, de mémoire ; [GN74] ne le cite pas).

-   *[GN74]* (texte lu) montrent par simulation que régresser des marches aléatoires indépendantes donne souvent un \$R\^2\$ élevé, une statistique de Durbin–Watson faible et des rejets fréquents ; ils ne démontrent pas l'asymptotique.

-   *[NK81]* (résumé lu) montrent qu'enlever par régression une tendance déterministe à une série qui est en fait une marche aléatoire fabrique des cycles apparents.

-   *[Phi86]* (résumé lu) donne l'asymptotique exacte : lois limites exprimées par des fonctionnelles browniennes, divergence de \$t\$, convergence de Durbin–Watson vers 0, \$R\^2\$ non dégénéré.

-   *[DP88]* (résumé lu ; ordre des auteurs non confirmé par la fiche) étendent ces fonctionnelles à la comparaison tendances contre marches aléatoires.

-   *[EG87]* (résumé lu) donnent le critère qui rend une régression entre deux séries intégrées *non* fallacieuse : la cointégration, c'est-à-dire l'existence d'une combinaison linéaire stationnaire.

**Esquisse de preuve** (pour une régression sur une tendance ; démonstration complète : [T5](00b_annexe_technique.md#T5)). Sous le principe d'invariance de Donsker (non vérifié ici), \$S\_{\\lfloor nr\\rfloor}/(\\sigma\\sqrt n)\\Rightarrow W(r)\$, mouvement brownien. Alors, par application continue : \$\$\\frac{\\hat b}{\\sqrt n}\\Rightarrow12\\sigma\\int_0\^1!\\big(r-\\tfrac12\\big)W(r),dr\\sim N\\big(0,\\tfrac65\\sigma\^2\\big),\\qquad \\frac{\\hat s\^2}{n}\\Rightarrow\\sigma\^2V,\\qquad \\frac{t_n}{\\sqrt n}\\Rightarrow\\frac{\\sqrt{12}\\int_0\^1(r-\\frac12)W}{\\sqrt V},\$\$ où \$V=\\int_0\^1(W-\\bar W)\^2-12\\big(\\int_0\^1(r-\\tfrac12)W\\big)\^2\>0\$ p.s. La limite de \$t_n/\\sqrt n\$ est non dégénérée, donc \$\|t_n\|\\to\\infty\$ à la vitesse \$\\sqrt n\$ ; de même \$n\\cdot\\mathrm{DW}\_n\\Rightarrow1/V\$.

**Contrôle numérique (E3-D).** 1000 marches aléatoires par taille, régression sur le temps, statistique \$t\$ « iid » :

| \$n\$ | \$\\mathbb P(\\lvert t\\rvert\>1.96)\$ | médiane de \$\\lvert t\\rvert/\\sqrt n\$ | médiane de \$n\\cdot\\mathrm{DW}\$ | \$\\operatorname{Var}(\\hat b/\\sqrt n)\$ (théorie 1.2) |
|----|----|----|----|----|
| 100 | 0.871 | 0.81 | 17.9 | 1.16 |
| 400 | 0.935 | 0.86 | 18.6 | 1.24 |
| 1600 | 0.974 | 0.93 | 17.6 | 1.28 |
| 6400 | 0.990 | 0.88 | 18.7 | 1.24 |

Le taux de rejet tend vers 1 ; \$\\lvert t\\rvert/\\sqrt n\$ et \$n\\cdot\\mathrm{DW}\$ se stabilisent ; \$\\operatorname{Var}(\\hat b/\\sqrt n)\$ reste proche de \$6/5\$ (erreur-type d'environ 0.05 avec 1000 répliques). Cela illustre l'énoncé pour une régression sur le temps ; cela ne démontre rien pour deux marches aléatoires indépendantes (cas de [GN74] et [Phi86]), qui suit les mêmes fonctionnelles mais n'est pas calculé ici.

**Que faire.** (i) Reconnaître le symptôme : \$R\^2\$ élevé, résidus quasi non décroissants (autocorrélation proche de 1), Durbin–Watson faible. (ii) Différencier : \$\\Delta S_t=\\varepsilon_t\$ est i.i.d., et la régression de \$\\Delta X\$ sur une constante n'est plus fallacieuse. (iii) Pour deux séries intégrées, tester la cointégration [EG87]. (iv) Ne pas se contenter de Newey–West : en E3-A, à \$n=500\$ il rejette encore 0.767 des marches aléatoires (théorie : sa statistique a une loi limite non standard, non démontrée ici).

### 6.4 Tests aux hypothèses nulles opposées

ADF et KPSS sont des *diagnostics* : ils ne pallient pas la dépendance AR, ils décident de la classe de modèle (différencier ou non). Description et lecture croisée : [P10](00b_annexe_technique.md#P10).

-   **ADF** (Dickey–Fuller augmenté [SD84]) : \$H_0\$ = racine unitaire ; loi non standard (fonctionnelle brownienne, [DF79] pour le cas sans retards) ; les retards absorbent la dépendance de type ARMA. Peu puissant près de la racine unitaire (E3-B : 22 % de rejets à \$\\varphi=0.98\$).

-   **KPSS** [KPSS92] : \$H_0\$ = stationnarité autour d'une tendance. Dans le modèle \$x_t=\\xi t+r_t+\\varepsilon_t\$, \$r_t=r\_{t-1}+u_t\$, c'est le test du multiplicateur de Lagrange de \$\\sigma_u\^2=0\$ (contenu non relu) : il teste que la composante aléatoire a une variance nulle, ce qui relie ce paragraphe au modèle de niveau local de [P5](00b_annexe_technique.md#P5). Les auteurs notent la faible puissance des tests usuels de racine unitaire (introduction lue) ; en E3-B, KPSS rejette à tort la stationnarité dans 85 % des cas à \$\\varphi=0.98\$ (taille nominale : 5 %).

-   **Ruptures.** Une rupture déterministe non modélisée est lue comme une composante aléatoire : les tests de racine unitaire ne rejettent plus, même asymptotiquement [Per89] (résumé lu) ; E3-B le montre (ADF : 51 % ; KPSS : 100 %).

### 6.5 Le critère qui tranche : la variance de l'erreur de prévision

> **Idée 9 — borné ou croissant.** Sous (D) et paramètres connus, l'erreur de prévision à \$h\$ pas a une variance bornée par \$\\gamma(0)\$. Il s'y ajoute l'erreur d'estimation de la droite, de variance \$\\approx\\tfrac{2\\pi f_S(0)}{n}\\big[1+12(\\tfrac12+\\tfrac hn)\^2\\big]\$ (droite ajustée sur \$n\$ points et prolongée de \$h\$ pas ; variance de long terme du §6.2), petite devant \$\\gamma(0)\$ tant que \$h\\ll n\$. Sous (R), la variance vaut \$h\\sigma\^2\$ et croît sans borne. Pour un AR(1), \$\\sigma\^2(1-\\varphi\^{2h})/(1-\\varphi\^2)\\approx h\\sigma\^2\$ tant que \$h(1-\\varphi)\\ll1\$ : les deux modèles donnent les mêmes intervalles pour \$h\$ petit devant la longueur de corrélation \$1/(1-\\varphi)\$ et se séparent au-delà. Se tromper ne coûte donc rien à court terme et beaucoup à long terme, ce que seule une évaluation à grand \$h\$, sur série longue, révèle. Dans le modèle de niveau local ([P5](00b_annexe_technique.md#P5)), la variance d'erreur \$P+h\\sigma\_\\eta\^2+\\sigma\_\\varepsilon\^2\$ croît linéairement dès que \$\\sigma\_\\eta\^2\>0\$.

Asymétrie : différencier à tort (la marche aléatoire pour une tendance stationnaire) est sûr mais produit des intervalles inutilement larges ; ne pas différencier à tort sous-couvre.

### 6.6 Ce que montre E3 (\$n=500\$)

-   **E3-A.** Taux de rejet de « pente nulle » à 5 % alors que la pente est nulle (2000 répliques) : voir §8. Le test iid suit la théorie \$2{1-\\Phi(1.96/\\sqrt r)}\$ ; Newey–West (\$L=22=\\lfloor\\sqrt n\\rfloor\$) est encore libéral, y compris à \$\\varphi=0\$ (8.1 %) et près de la racine unitaire (46 % à 0.98). Pour la marche aléatoire, le rejet vaut 0.866, 0.949, 0.970 à \$n=100,500,2000\$.

-   **E3-B.** ADF et KPSS (150 répliques) : corrects aux extrêmes, se contredisent dans la zone grise (tendance + AR(0.98) : ADF rejette la racine unitaire 22 % du temps, KPSS rejette la stationnarité 85 % du temps) et sont trompés par une rupture de niveau.

-   **E3-C.** Deux recettes (tendance + AR(1) ; marche aléatoire avec dérive) contre deux vérités, à \$h=1,24,100\$ (1000 répliques par vérité) : voir §8.

-   **E3-D.** Divergence en \$\\sqrt n\$ de la statistique \$t\$ sous marche aléatoire (§6.3).

## 7. Premières limites

1.  **Un seul jeu de paramètres par scénario** (\$\\varphi=0.85\$, \$\\theta=0.30\$ en E1 ; \$\\varphi=0.7\$ et \$(\\omega,a,b)=(0.05,0.10,0.85)\$ en E2 ; etc.), sans étude de sensibilité. Les ordres de grandeur sont fiables ; les seuils (p. ex. le croisement persistance/climatologie à \$h=2\$–\$3\$) dépendent de \$\\varphi\$ et de l'amplitude du cycle.

2.  **\$R=20\$ répliques** en E1 et E2 : erreurs-types jusqu'à 0.018 en MASE. Les écarts de l'ordre de 0.01 (climatologie + AR contre oracle) ne sont pas significatifs.

3.  **E2 favorise le GARCH par construction** : le vrai modèle est un GARCH gaussien. L'objectif est de montrer ce que couvre chaque garantie, pas de classer les méthodes.

4.  **Intervalles à \$h=1\$ seulement** (E2, E2b). À \$h\\ge2\$ la loi prédictive du GARCH est un mélange d'échelles : non évalué.

5.  **L'oracle de E2c connaît l'état caché** : c'est un « génie » qui utilise plus d'information que les données (ensemble \$\\mathcal G_t\\supsetneq\\mathcal F_t\$). Le filtre de Hamilton n'est pas implémenté ; je n'ai pas quantifié l'écart à l'optimum \$\\mathcal F_t\$-mesurable.

6.  **Les EDS ne sont pas évaluées** : générateur et vérification seulement (raisons au §3.4).

7.  **Tests de E3** : \$n=500\$, ADF avec choix de retards par AIC, KPSS avec fenêtre automatique, Newey–West avec \$L=\\lfloor\\sqrt n\\rfloor\$ ; d'autres choix changent les taux.

8.  **Aucune donnée réelle**, aucune marginale non gaussienne en E1 et E3, pas de valeurs manquantes, pas de variance saisonnière, pas de mémoire longue.

9.  **Aucun réseau de neurones n'est entraîné** : le §11 énonce des correspondances ; seuls les points accompagnés d'une sortie de cellule (pertes, jacobienne, stratégie directe ou récursive) sont vérifiés.

10. **Exécution** : voir §10.

**Grille de lecture pour Blagnac** (sans décision) : rapporter le MASE en fonction de \$h\$ et non à un seul horizon (E1) ; rapporter la couverture conditionnelle (groupes de volatilité, heure du jour) en plus de la marginale (E2) ; tester la tenue après un changement de variance entre calibration et test (E2b) ; vérifier la forme de la loi résiduelle et la couverture à plusieurs niveaux, pas seulement à 90 % (E2c) ; traiter la tendance comme un choix de modèle contrôlé à grand horizon (E3).

## 8. Résultats observés

Graine de `tsunc.config` (\$42\$). Moyennes de \$R=20\$ répliques (E1, E2, E2b).

**E1. MASE** (échelle : naïve saisonnière sur l'apprentissage ; erreur-type maximale 0.018).

|                           | \$h=1\$ | \$h=6\$ | \$h=12\$ | \$h=24\$ |
|---------------------------|---------|---------|----------|----------|
| persistance               | 0.366   | 1.251   | 1.628    | 0.999    |
| naïve saisonnière         | 0.998   | 0.999   | 0.999    | 0.999    |
| climatologie              | 0.718   | 0.718   | 0.718    | 0.718    |
| climatologie + AR(3)      | 0.296   | 0.656   | 0.711    | 0.718    |
| oracle (vrais paramètres) | 0.296   | 0.652   | 0.706    | 0.714    |
| théorie : oracle          | 0.298   | 0.654   | 0.707    | 0.715    |
| théorie : climatologie    | 0.715   | 0.715   | 0.715    | 0.715    |

Persistance : 0.366, 0.608, 0.802, 0.971 à \$h=1,2,3,4\$, contre 0.718 pour la climatologie.

**E2. Intervalles à 90 %, un pas** (1500 heures de test par réplique).

|   | couverture | faible vol. | vol. moyenne | forte vol. | largeur moy. | score d'intervalle |
|----|----|----|----|----|----|----|
| gaussien, variance constante | 0.903 | 0.964 | 0.921 | 0.823 | 3.269 | 4.267 |
| conforme séparé | 0.904 | 0.965 | 0.922 | 0.826 | 3.298 | 4.276 |
| régression quantile + proxy | 0.902 | 0.920 | 0.907 | 0.878 | 3.205 | 4.079 |
| GARCH(1,1), quasi-vraisemblance | 0.901 | 0.903 | 0.902 | 0.898 | 3.202 | 4.003 |
| oracle (\$\\sigma_t\$ connu) | 0.901 | 0.902 | 0.901 | 0.900 | 3.205 | 3.997 |

Les colonnes de volatilité sont les terciles de la vraie \$\\sigma_t\$.

**E2b. Couverture après une rupture \$\\sigma\\times2\$** (cible 0.90).

|   | 250 premières h | 250–750 h | 750–1500 h | bloc entier |
|----|----|----|----|----|
| gaussien, variance constante | 0.577 | 0.625 | 0.608 | 0.608 |
| conforme séparé | 0.569 | 0.618 | 0.599 | 0.600 |
| conforme glissant (\$W=500\$) | 0.699 | 0.909 | 0.890 | 0.865 |
| GARCH(1,1) | 0.812 | 0.834 | 0.826 | 0.826 |

Après la rupture, la variance filtrée par GARCH vaut en moyenne 2.995 (point fixe \$(\\omega+4a)/(1-b)=3\$), la variance conditionnelle vraie 4.026. À écart-type uniformément sous-estimé d'un facteur \$\\sqrt{3/4}\$, la couverture gaussienne serait 0.846 (observé : 0.83).

**E2c. Deux régimes** (une trajectoire de 60 000 points, moitié apprentissage).

| \$h\$ | niveau | méthode                    | couverture | largeur moy. |
|-------|--------|----------------------------|------------|--------------|
| 1     | 0.5    | gaussien AR                | 0.533      | 2.079        |
| 1     | 0.5    | oracle (connaît le régime) | 0.504      | 1.438        |
| 1     | 0.9    | gaussien AR                | 0.905      | 5.070        |
| 1     | 0.9    | oracle                     | 0.900      | 4.239        |
| 24    | 0.5    | gaussien AR                | 0.316      | 3.006        |
| 24    | 0.5    | oracle                     | 0.505      | 3.984        |
| 24    | 0.9    | gaussien AR                | 0.953      | 7.331        |
| 24    | 0.9    | oracle                     | 0.903      | 6.558        |

Lecture : à \$h=24\$ les valeurs du gaussien AR sont celles de \$N(0,5)\$ calculées au §3.3 (0.311 et 0.953 ; largeurs 3.016 et 7.356) : la loi prédictive est mal représentée, non par une erreur d'estimation.

**E3-A. « Pente significative » à 5 % alors que la pente est nulle** (\$n=500\$, 2000 répliques).

| \$\\varphi\$ | théorie, écart-type iid | rejets, écart-type iid | rejets, Newey–West |
|----|----|----|----|
| 0 | 0.050 | 0.045 | 0.081 |
| 0.5 | 0.258 | 0.244 | 0.084 |
| 0.9 | 0.653 | 0.646 | 0.177 |
| 0.98 | 0.844 | 0.848 | 0.461 |
| 1 (marche aléatoire) | tend vers 1 | 0.947 | 0.767 |

**E3-B. Part des répliques où chaque test rejette son hypothèse nulle** (5 %, \$n=500\$, 150 répliques).

|   | ADF rejette la racine unitaire | KPSS rejette la stationnarité |
|----|----|----|
| tendance + AR(0.7) | 1.000 | 0.087 |
| marche aléatoire + dérive | 0.047 | 0.953 |
| tendance + AR(0.98) | 0.220 | 0.847 |
| tendance + rupture de niveau, AR(0.7) | 0.507 | 1.000 |

**E3-C. Intervalles à 90 %** (1000 répliques par vérité). La couverture à \$h\$ est la fréquence avec laquelle la valeur observée à \$t+h\$ tombe dans l'intervalle ; la largeur est la largeur moyenne de l'intervalle.

| vérité | recette | couv. \$h=1\$ | \$h=24\$ | \$h=100\$ | largeur \$h=1\$ | \$h=24\$ | \$h=100\$ |
|----|----|----|----|----|----|----|----|
| marche aléatoire + dérive | marche aléatoire + dérive | 0.905 | 0.895 | 0.861 | 3.28 | 16.09 | 32.84 |
| marche aléatoire + dérive | tendance + AR(1) | 0.902 | 0.753 | 0.504 | 3.27 | 12.99 | 17.34 |
| tendance + AR(0.7) | marche aléatoire + dérive | 0.897 | 1.000 | 1.000 | 3.57 | 17.51 | 35.75 |
| tendance + AR(0.7) | tendance + AR(1) | 0.901 | 0.884 | 0.870 | 3.29 | 4.55 | 4.55 |

La couverture de 0.870 à \$h=100\$ sous la bonne recette n'est pas décomposée. L'incertitude sur la pente ajoute à la variance d'erreur environ \$\\frac{2\\pi f_S(0)}{n}\\big[1+12(\\tfrac12+\\tfrac hn)\^2\\big]\\approx0.15\$ (\$\\varphi=0.7\$, \$n=500\$, \$h=100\$), soit 8 % de \$\\gamma(0)\$ : la couverture gaussienne passerait de 0.90 à 0.89. Le reste de l'écart (estimation de \$\\varphi\$ et de la variance résiduelle) n'est pas quantifié ; aucune des deux recettes ne propage l'incertitude sur les paramètres.

**E3-D. Régression fallacieuse, asymptotique** : tableau du §6.3.

**NN-A/B/C. Pont vers les réseaux** (§11) : tableaux du §11.

## 9. Références

Colonnes : l'étiquette, la référence, ce que la source établit **pour ce document**, le statut de lecture (légende en tête).

| Étiquette | Référence | Ce que la source établit | Statut |
|----|----|----|----|
| [BCRT23] | Barber, R. F., Candès, E. J., Ramdas, A. et Tibshirani, R. J. (2023). Conformal prediction beyond exchangeability. *Ann. Statist.* 51(2), 816–845. <doi:10.1214/23-AOS2276> | Garanties de couverture sans échangeabilité, par quantiles pondérés et randomisation ; le cas glissant comme cas particulier n'est pas vérifié | ✔ résumé lu |
| [BSB05] | Benth, F. E. et Šaltytė Benth, J. (2005). Stochastic modelling of temperature variations with a view towards weather derivatives. *Appl. Math. Finance*12(1), 53–85 | Température : moyenne et volatilité saisonnières, bruit de type OU piloté par un processus de Lévy hyperbolique généralisé | ✔ résumé lu |
| [BD91] | Brockwell, P. J. et Davis, R. A. *Time Series: Theory and Methods*, 2e éd. Springer | Ouvrage de référence (causalité, inversibilité, Wold) ; énoncés de T1 non comparés au texte | ◐ ouvrage non relu |
| [BJRL15] | Box, G. E. P., Jenkins, G. M., Reinsel, G. C. et Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*, 5e éd. Wiley | Ouvrage de référence sur ARIMA et SARIMA | ◐ ouvrage non relu |
| [Bol86] | Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *J. Econometrics*31(3), 307–327 | Définit le GARCH ; conditions de stationnarité et structure d'autocorrélation. Le kurtosis et la forme ARMA du carré sont redémontrés en T3, non comparés au texte | ✔ résumé lu |
| [CD05] | Campbell, S. D. et Diebold, F. X. (2005). Weather forecasting for weather derivatives. *J. Amer. Statist. Assoc.* 100, 6–16 (NBER w10141) | Dynamique de la moyenne et de la variance conditionnelles de la température quotidienne | ✔ résumé lu |
| [DF79] | Dickey, D. A. et Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *J. Amer. Statist. Assoc.* 74, 427–431 (volume et pages de mémoire) | Lois limites de l'estimateur et du test \$t\$ sous \$\\rho=1\$ | ◐ titre et résumé lus ; volume et pages non vérifiés |
| [DK12] | Durbin, J. et Koopman, S. J. *Time Series Analysis by State Space Methods*, 2e éd. Oxford University Press | Ouvrage de référence sur l'espace d'état et le filtre de Kalman | ◐ ouvrage non relu |
| [DP88] | Durlauf, S. N. et Phillips, P. C. B. (1988). Trends versus random walks in time series analysis. *Econometrica* 56(6), 1333–1354 | Fonctionnelles browniennes pour la comparaison tendances et marches aléatoires | ◐ résumé lu ; ordre des auteurs non confirmé par la fiche |
| [EG87] | Engle, R. F. et Granger, C. W. J. (1987). Co-integration and error correction: representation, estimation, and testing. *Econometrica* 55(2), 251–276 (pages de mémoire) | Cointégration : combinaison stationnaire de séries intégrées | ◐ résumé lu ; pages non vérifiées |
| [Eng82] | Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica* 50(4), 987–1007 | Introduit l'ARCH, appliqué à la variance de l'inflation britannique | ✔ résumé lu |
| [GBR07] | Gneiting, T., Balabdaoui, F. et Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness. *J. R. Stat. Soc. B* 69(2), 243–268 | Distingue calibration et finesse ; maximiser la finesse sous calibration | ✔ résumé lu |
| [GC21] | Gibbs, I. et Candès, E. (2021). Adaptive conformal inference under distribution shift. NeurIPS 34 ; arXiv:2106.00170 | ACI : ajustement en ligne du niveau, couverture de long terme malgré les changements de loi ; règle de mise à jour non relue | ✔ résumé lu |
| [GJ80] | Granger, C. W. J. et Joyeux, R. (1980). An introduction to long-memory time series models and fractional differencing. *J. Time Ser. Anal.* 1(1), 15–29 | Modèles à mémoire longue, différenciation fractionnaire | ◐ fiche seule |
| [GN74] | Granger, C. W. J. et Newbold, P. (1974). Spurious regressions in econometrics. *J. Econometrics* 2(2), 111–120 | Simulations : régressions de marches aléatoires indépendantes, \$R\^2\$ élevé, DW faible, rejets fréquents ; ne démontre pas l'asymptotique ; ne cite pas Yule (1926) | ✔ texte lu |
| [GR07] | Gneiting, T. et Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *J. Amer. Statist. Assoc.* 102(477), 359–378. <doi:10.1198/016214506000001437> | Propreté des règles (pinball, score d'intervalle) ; CRPS comme intégrale de scores de Brier ; orientation du score non confirmée dans le texte publié | ✔ texte lu (version rapport technique) |
| [Ham89] | Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica* 57(2), 357–384 | Régimes de Markov appliqués au cycle économique | ✔ résumé lu |
| [HA18] | Hyndman, R. J. et Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*, 2e éd. OTexts, §3.4 | Échelle du MASE (erreur de la naïve) | ✔ texte lu (§3.4) |
| [HK06] | Hyndman, R. J. et Koehler, A. B. (2006). Another look at measures of forecast accuracy. *Int. J. Forecasting*. <doi:10.1016/j.ijforecast.2006.03.001> | Définit le MASE | ◐ fiche seule |
| [Hos81] | Hosking, J. R. M. (1981). Fractional differencing. *Biometrika* 68(1), 165–176 | Fonction d'autocorrélation de l'ARFIMA ; constante exacte à relire | ◐ fiche seule (auteur non affiché) |
| [HS97] | Hochreiter, S. et Schmidhuber, J. (1997). Long short-term memory. *Neural Computation* 9(8), 1735–1780 | Cellule à flux d'erreur constant contre le gradient évanescent | ✔ résumé lu |
| [Bis94] | Bishop, C. M. (1994). Mixture density networks. Rapport technique NCRG/94/004, Aston University | Réseau dont la sortie est un mélange de lois | ◐ fiche seule |
| [BSF94] | Bengio, Y., Simard, P. et Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. *IEEE Trans. Neural Networks*5(2), 157–166 | Difficulté de l'apprentissage de dépendances longues par descente de gradient (d'après le titre) | ◐ fiche seule |
| [KB78] | Koenker, R. et Bassett, G. (1978). Regression quantiles. *Econometrica* 46(1) | Définit la régression quantile (la preuve du minimiseur est donnée ici) | ◐ fiche seule ; pages non vérifiées |
| [KP92] | Kloeden, P. E. et Platen, E. *Numerical Solution of Stochastic Differential Equations*. Springer | Ouvrage de référence sur Euler–Maruyama | ◐ ouvrage non relu |
| [KPSS92] | Kwiatkowski, D., Phillips, P. C. B., Schmidt, P. et Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. *J. Econometrics* 54, 159–178 | \$H_0\$ : stationnarité ; faible puissance des tests usuels de racine unitaire (introduction) | ✔ texte lu (introduction) |
| [KSC98] | Kim, S., Shephard, N. et Chib, S. (1998). Stochastic volatility: likelihood inference and comparison with ARCH models. *Rev. Econ. Stud.*65(3), 361–393 | Inférence par vraisemblance pour la volatilité stochastique | ◐ fiche seule ; notice non reconfirmée |
| [KV05] | Kiefer, N. M. et Vogelsang, T. J. (2005). A new asymptotic theory for heteroskedasticity-autocorrelation robust tests. *Econometric Theory*21(6), 1130–1164 | Distorsions de taille des tests HAC selon la fenêtre | ✔ résumé lu |
| [LGRTW18] | Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. et Wasserman, L. (2018). Distribution-free predictive inference for regression. *J. Amer. Statist. Assoc.*113(523), 1094–1111 | Couverture en échantillon fini pour tout estimateur de régression (conforme séparé) | ✔ résumé lu |
| [NK81] | Nelson, C. R. et Kang, H. (1981). Spurious periodicity in inappropriately detrended time series. *Econometrica* 49(3), 741–751 (pages de mémoire) | Retirer une tendance déterministe à une marche aléatoire crée des cycles apparents | ◐ résumé lu ; pages non vérifiées |
| [NP82] | Nelson, C. R. et Plosser, C. I. (1982). Trends and random walks in macroeconomic time series. *J. Monetary Econ.* 10(2), 139–162 | Pose la question tendance déterministe ou stochastique pour des séries macroéconomiques | ✔ résumé lu |
| [NW87] | Newey, W. K. et West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica* 55(3), 703–708 | Estimateur HAC semi-défini positif et convergent ; poids de Bartlett non relus | ✔ résumé lu |
| [Per89] | Perron, P. (1989). The Great Crash, the oil price shock, and the unit root hypothesis. *Econometrica*57(6), 1361–1401 | Sous une rupture déterministe, les tests de racine unitaire ne rejettent pas, même asymptotiquement | ✔ résumé lu |
| [Phi86] | Phillips, P. C. B. (1986). Understanding spurious regressions in econometrics. *J. Econometrics*33(3), 311–340 | Asymptotique de la régression fallacieuse (fonctionnelles browniennes) | ✔ résumé lu |
| [Ric81] | Richardson, C. W. (1981). Stochastic simulation of daily precipitation, temperature, and solar radiation. *Water Resour. Res.* 17(1), 182–190. <doi:10.1029/wr017i001p00182> | Générateur de temps | ◐ fiche seule |
| [RPC19] | Romano, Y., Patterson, E. et Candès, E. (2019). Conformalized quantile regression. NeurIPS 32 ; arXiv:1905.03222 | CQR : conforme appliqué à la régression quantile | ✔ résumé lu |
| [RW06] | Rasmussen, C. E. et Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press | Ouvrage de référence sur les noyaux, dont Matérn | ◐ ouvrage non relu (site consulté) |
| [Sal20] | Salinas, D., Flunkert, V., Gasthaus, J. et Januschowski, T. (2020). DeepAR: probabilistic forecasting with autoregressive recurrent networks. *Int. J. Forecasting* 36(3), 1181–1191 (volume et pages de mémoire) | Réseau récurrent à sortie probabiliste, entraîné par vraisemblance | ◐ fiche seule ; volume et pages non vérifiés |
| [SD84] | Said, S. E. et Dickey, D. A. (1984). Testing for unit roots in autoregressive-moving average models of unknown order. *Biometrika* 71(3), 599–607 | ADF : approximation d'un ARMA par un autorégressif ; retards ajoutés | ◐ résumé lu ; notice non reconfirmée |
| [SSKKEP21] | Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S. et Poole, B. (2021). Score-based generative modeling through stochastic differential equations. ICLR 2021 ; arXiv:2011.13456 | EDS directe vers le bruit, EDS rétrograde exigeant le score ; forme exacte de la EDS « VP » non vérifiée | ✔ résumé lu |
| [Tay00] | Taylor, J. W. (2000). A quantile regression neural network approach to estimating the conditional density of multiperiod returns. *J. Forecasting*19(4), 299–311 | Réseau à sortie quantile (d'après le titre) | ◐ fiche seule |
| [TL80] | Tong, H. et Lim, K. S. (1980). Threshold autoregression, limit cycles and cyclical data. *J. R. Stat. Soc. B* 42(3), 245–292 | Modèles autorégressifs à seuil | ◐ fiche seule |
| [UO30] | Uhlenbeck, G. E. et Ornstein, L. S. (1930). On the theory of the Brownian motion. *Phys. Rev.* 36, 823 | Théorie du mouvement brownien (OU) | ◐ fiche seule (page finale non lue) |
| [WW99] | Wilks, D. S. et Wilby, R. L. (1999). The weather generation game: a review of stochastic weather models. *Prog. Phys. Geogr.* 23(3), 329–357 | Revue des générateurs de temps | ◐ fiche seule |
| [Zen23] | Zeng, A., Chen, M., Zhang, L. et Xu, Q. (2023). Are transformers effective for time series forecasting? AAAI 2023 ; arXiv:2205.13504 | Un modèle linéaire simple rivalise avec des Transformers sur la prévision à long horizon (résumé) | ✔ résumé lu |

Référence de contexte, non citée dans le texte : Murphy, A. H. (1992). Climatology, persistence, and their linear combination as standards of reference in skill scores. *Weather and Forecasting* 7(4), 692–698 (fiche lue ; contenu non lu).

**Repères historiques et résultats classiques cités de mémoire, non vérifiés** : Legendre et Gauss (moindres carrés, vers 1805–1809) ; Langevin (1908) ; Yule (1926, corrélations absurdes ; 1927, AR(2) et taches solaires) ; Slutsky (1937) ; Wold (1938, décomposition) ; Kramers (1940) ; Hurst (1951) ; Hasselmann (1976) ; Vasicek (1977) ; Benzi, Sutera et Vulpiani (1982) ; théorème ergodique de Birkhoff ; principe d'invariance de Donsker ; cohérence de la quasi-vraisemblance GARCH ; identité du CRPS ; formule exacte de \$\\rho(h)\$ pour l'ARFIMA ; forme de la statistique KPSS.

## 10. Reproduire

Notebook : [`00b_synthetic_methods.ipynb`](../notebooks/00b_synthetic_methods.ipynb) — [ouvrir dans Colab](https://colab.research.google.com/github/GalaxiLir/stats-to-deeplearning/blob/main/01-ts-uncertainty/notebooks/00b_synthetic_methods.ipynb) (le lien ne fonctionne qu'une fois le fichier poussé). Durée : environ une minute. Annexe : [`00b_annexe_technique.md`](00b_annexe_technique.md).

Dépendances : `numpy`, `pandas`, `matplotlib` (déjà dans `pyproject.toml`) et `scipy`, `scikit-learn`, `statsmodels`(préinstallés sur Colab ; absents de `pyproject.toml` et de `environment.yml`, à y ajouter pour le travail local). La première cellule installe ce qui manque (ces trois paquets, puis `tsunc` depuis GitHub) ; le notebook réutilise `tsunc.config` (`ALPHA`, `SEED`, `set_seed`) et `tsunc.metrics` (`pinball_loss`, `mase`, `coverage`, `mean_width`). Il définit une seule fonction de plus, `interval_score`, qui enveloppe `pinball_loss`. Les nouvelles cellules (E3-D, pont vers les réseaux) n'utilisent que `numpy`, `pandas`, `scipy` et les fonctions déjà définies plus haut ; aucun réseau n'est entraîné, `torch`n'est pas requis.

Version d'exécution : Python 3.13, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, scikit-learn 1.9.1, statsmodels 0.15.0, matplotlib 3.11.2.

**Exécution.** La version du 2026-11-18 (48 cellules, dont 5 nouvelles cellules de code : E3-D et les quatre du §11) a été exécutée de bout en bout, sans erreur, en 45 s, dans l'environnement de développement ci-dessus. Les sorties textuelles des 18 cellules déjà présentes le 16/11 sont identiques à celles de la version précédente. Cette version précédente avait aussi été exécutée dans un environnement vierge (installation de `scipy`, `scikit-learn`, `statsmodels`, `tsunc`) et dans un environnement à pile plus ancienne (numpy 2.1.3, pandas 2.2.3, scipy 1.14.1, scikit-learn 1.6.1, statsmodels 0.14.4, matplotlib 3.10.0), avec des sorties identiques ; la version du 18/11 n'a pas été réexécutée dans ces deux environnements. Elle n'a pas été exécutée sur Colab même (image, version de Python et paquets propres à Colab).

## 11. Et les réseaux de neurones ?

Le fil conducteur du projet est le passage de ces méthodes à des réseaux de neurones. Ce paragraphe énonce les correspondances, les teste là où une cellule du notebook le permet, et signale ce qui reste une attente.

### 11.1 Même cible, mêmes pertes, même plancher

Un réseau est une famille paramétrique \$f\_\\theta\$ (de la fenêtre passée vers la prévision) ajustée par minimisation d'un risque empirique \$\\hat\\theta=\\arg\\min\_\\theta n\^{-1}\\sum_i\\ell(y_i,f\_\\theta(x_i))\$. Par l'Idée 1, **la perte choisit la fonctionnelle estimée** : erreur quadratique \$\\to\$ espérance conditionnelle ; pinball \$\\to\$ quantile conditionnel (réseau à sortie quantile, [Tay00]) ; log-vraisemblance négative \$\\to\$ densité (réseaux à sortie probabiliste, [Bis94 ; Sal20]).

*Test (NN-A).* Sur 200 000 tirages d'une loi log-normale, on minimise numériquement chaque perte sur une constante :

| perte | argmin du risque empirique | fonctionnelle de l'échantillon | fonctionnelle de la loi |
|----|----|----|----|
| quadratique \$\\to\$ moyenne | 1.377 | 1.377 | 1.377 |
| pinball 0.5 \$\\to\$ médiane | 0.999 | 0.999 | 1.000 |
| pinball 0.9 \$\\to\$ quantile 0.9 | 2.780 | 2.780 | 2.788 |

Les deux premières colonnes coïncident (c'est l'Idée 1 appliquée à la loi empirique) ; l'écart à la troisième (2.780 contre 2.788) est l'erreur d'échantillonnage. Cela montre que la perte sélectionne la fonctionnelle ; cela ne dit rien d'un réseau.

**Le plancher de l'oracle.** Les sections E1 et E2 donnent, pour chaque série, le risque minimal atteignable avec l'information \$\\mathcal F_t\$ (par exemple MASE de l'oracle 0.296 à \$h=1\$ en E1). Un réseau qui bat l'oracle *à information égale* au-delà de l'erreur-type signale une fuite (§1.1, H4) ou un sur-ajustement du bloc de test, car l'oracle est optimal pour une perte propre. À information plus grande (E2c : l'oracle connaît l'état caché), la comparaison n'a pas ce sens.

### 11.2 Un AR(1) est un réseau récurrent linéaire ; le gradient évanescent est la mémoire géométrique

La récurrence \$h_t=\\varphi h\_{t-1}+x_t\$ est une cellule récurrente linéaire à une unité, de poids \$\\varphi\$. Par la règle de dérivation en chaîne, \$\\partial h_T/\\partial h\_{T-k}=\\varphi\^k\$ : c'est la jacobienne que la rétropropagation à travers le temps multiplie \$k\$ fois. Pour un AR(1), c'est aussi le poids de Wold \$\\psi_k=\\varphi\^k\$ et l'autocorrélation \$\\rho(k)\$ ([T2](00b_annexe_technique.md#T2)) : **la mémoire statistique et la portée du gradient sont le même objet**. Pour \$\|\\varphi\|\<1\$, le gradient décroît géométriquement (gradient évanescent) ; pour \$\\varphi=1\$, il se conserve (c'est le flux d'erreur constant de [HS97] ; [BSF94] analysent la difficulté de l'apprentissage de dépendances longues par descente de gradient, d'après le titre). Pour une cellule non linéaire \$h_t=\\tanh(wh\_{t-1}+\\dots)\$, la jacobienne est un produit de facteurs \$w(1-h_j\^2)\$, majoré par \$\|w\|\^k\$ : l'évanescence est la même mécanique (heuristique ; non vérifié dans [BSF94]). Par ailleurs, la reconstruction du bruit d'un MA(1) par \$\\hat\\varepsilon_t=X_t-\\theta\\hat\\varepsilon\_{t-1}\$ est une cellule récurrente de poids \$-\\theta\$ : l'inversibilité \$\|\\theta\|\<1\$ en est la stabilité ([T1](00b_annexe_technique.md#T1)).

*Test (NN-B).* Jacobiennes par différences finies \$\\partial h_T/\\partial h\_{T-k}\$ sur la récurrence, comparées à \$\\varphi\^k\$ :

| \$\\varphi\$ | \$k=1\$ | \$k=10\$               | \$k=50\$ | \$k=100\$            |
|--------------|---------|------------------------|----------|----------------------|
| 0.50         | 0.50    | \$9.8\\times10\^{-4}\$ | 0.000    | 0.000                |
| 0.90         | 0.90    | 0.349                  | 0.005    | \$3\\times10\^{-5}\$ |
| 0.99         | 0.99    | 0.904                  | 0.605    | 0.366                |
| 1.00         | 1.00    | 1.000                  | 1.000    | 1.000                |

Les différences finies et \$\\varphi\^k\$ coïncident aux chiffres affichés. Pour \$\\varphi=0.9\$, le poids de Wold \$\\psi_k\$ vaut 0.349, 0.0052 et \$3\\times10\^{-5}\$ à \$k=10,50,100\$ et l'autocorrélation empirique (200 000 points) 0.354, 0.006, 0.004 (l'écart à \$k=100\$ est de l'ordre du bruit d'échantillonnage : l'écart-type de l'autocorrélation empirique d'un AR(1) de coefficient 0.9 à 200 000 points est d'environ 0.0065, obtenu par simulation de 100 trajectoires). Le gradient passe sous 5 % de sa valeur initiale au bout de 5, 29 et 299 pas pour \$\\varphi=0.5\$, \$0.9\$, \$0.99\$ (\$\\log0.05/\\log\\varphi\$). Cela prouve l'égalité jacobienne \$=\\psi_k\$ pour la récurrence linéaire ; cela ne dit rien du comportement de l'entraînement d'un réseau non linéaire.

**Direct ou récursif à plusieurs pas.** Un modèle à un pas, itéré \$h\$ fois, prévoit avec \$\\rho(1)\^h\$ ; une sortie *directe* à l'horizon \$h\$ (un coefficient par horizon, comme la « climatologie + AR direct » de E1) vise \$\\rho(h)\$. Les deux coïncident si le processus est un AR(1), pas sinon.

*Test (NN-C).* Chaîne de Markov à deux états du §3.3 (\$\\rho_X(h)=0.8\\cdot0.9\^h\$), 300 000 points :

|   | \$h=1\$ | \$h=6\$ | \$h=24\$ |
|----|----|----|----|
| direct : coefficient linéaire optimal (autocorrélation empirique) | 0.718 | 0.424 | 0.068 |
| théorie \$0.8\\times0.9\^h\$ | 0.720 | 0.425 | 0.064 |
| récursif : \$\\rho(1)\^h\$ | 0.718 | 0.137 | 0.0004 |

Le récursif sous-estime la mémoire d'un facteur 3 à \$h=6\$ et ne retient presque rien à \$h=24\$ : un décodage autorégressif d'un modèle à un pas oublie une mémoire qui n'est pas celle d'un AR(1). Ici, la mémoire vient d'un état latent, et un réseau à sortie par horizon (ou un état récurrent assez riche) évite cette perte.

### 11.3 Mélanges de Markov et réseaux à densité de mélange

À \$h=24\$, E2c montre que la loi prédictive est un mélange, mal représenté par une gaussienne (§3.3). Un *réseau à densité de mélange* [Bis94] sort des poids \$\\pi_k(x)\$, des moyennes \$\\mu_k(x)\$ et des échelles \$\\sigma_k(x)\$ ; entraîné par log-vraisemblance (Idée 1), il estime \$\\sum_k\\pi_k(x)N(\\mu_k(x),\\sigma_k(x)\^2)\$. Les poids attendus sont les probabilités filtrées propagées \$\\sum_i\\mathbb P(s_t=i\\mid\\mathcal F_t)(P\^h)\_{ik}\$ du filtre de Hamilton ([P4](00b_annexe_technique.md#P4)). **Attente, non testée** : un tel réseau devrait rattraper l'oracle de E2c au niveau 50 % là où le gaussien AR couvre 0.316.

### 11.4 OU, double puits et diffusion

L'OU est le processus de bruitage le plus simple des modèles génératifs à diffusion [SSKKEP21] (la forme exacte de leurs EDS n'a pas été relue). Si la loi initiale est \$N(m_0,v_0)\$, la transition de [P9](00b_annexe_technique.md#P9) donne une loi à l'instant \$t\$ gaussienne \$N(m_0e\^{-\\kappa t},,v_0e\^{-2\\kappa t}+\\sigma\^2\\frac{1-e\^{-2\\kappa t}}{2\\kappa})\$, de score linéaire \$-(x-m_t)/v_t\$ : le score est explicite, ce qui rend l'EDS rétrograde calculable. Pour le double puits, la loi à l'instant \$t\$ n'est pas gaussienne : le score doit être *appris* par un réseau, qui est l'objet de la reproduction « diffusion » du plan. Aucune cellule n'entraîne de score ici.

### 11.5 Limites et précautions

-   Rien dans ce paragraphe n'entraîne de réseau ; les trois tests (NN-A, NN-B, NN-C) portent sur des pertes, une jacobienne et une stratégie de prévision, pas sur de l'apprentissage.

-   **Incertitude.** Pour un réseau, il y a deux sources d'aléa : les données (déjà gérée par les \$R\$ trajectoires indépendantes) et l'initialisation et l'ordre des lots (graines). L'erreur-type doit couvrir les deux.

-   **Sélection d'hyperparamètres** : sur un bloc situé *entre* l'apprentissage et le test, jamais sur le test (H4).

-   **Lignes de base.** [Zen23] (résumé lu) rapportent qu'un modèle linéaire simple rivalise avec des Transformers sur la prévision à long horizon ; les méthodes de E1 doivent rester dans toute comparaison.

### 11.6 Programme du projet suivant (attentes, non testées)

1.  Réseau (perceptron multicouche, puis GRU) sur les générateurs de E1, E2, E2c, avec les pertes quadratique, pinball et log-vraisemblance ; comparaison à l'oracle et aux méthodes du §4, \$R\$ trajectoires et plusieurs graines.

2.  Attentes : en E1, le réseau rejoint « climatologie + AR » (plancher : oracle) sans le battre ; en E2, une sortie quantile conditionnelle réduit l'écart de couverture par tercile ; en E2c, une sortie à mélange rattrape l'oracle au niveau 50 %.

3.  Mise en place de la reproduction « diffusion » sur l'OU et le double puits (§11.4).
