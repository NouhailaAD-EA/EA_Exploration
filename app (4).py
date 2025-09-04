from pandas._libs.tslibs import period
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy import stats


df = pd.read_csv("data_evt_ea.csv")
df_age = pd.read_csv("data_age_cuve.csv")  # contient codpot, agebsq, date

# Sidebar pour choisir période

st.sidebar.title("Paramètres")
periode = st.sidebar.selectbox(
      "Choisir la période",
      ["18/06/2025 → 31/07/2025", "18/08/2025 → Aujourd'hui","Période initiale","Parties notes"]
  )
CUVES_PAR_GT = 33


if periode == "Parties notes":


  st.title("Analyse des cuves > 60 jours")

  # Compte rendu synthétique
  st.header("Compte rendu synthétique")

  st.markdown("""
  **1. Énergie et durée de polarisation**  
  - Faibles sur la période analysée → Une diminution petit à petit à travers les périodes .  
  - Attention : Les tests de référence n’étaient pas significatifs pour comparer les périodes.

  **2. Cause principale d’échec : Boucles max**  
  - Même après serrage, cette cause reste dominante.  
  - Indique une instabilité du procédé automatique plutôt qu’un problème lié à l’anode ou à l’opérateur.

  **3. Impact du serrage**  
  - Stabilise temporairement la cuve.  
  - Surutilisation possible si la boucle max se répète → Corrélation forte entre le serrage correctif et les causes d'echec boucle max => plus de serrage  .

  **4. Comparaison entre périodes**  
  - La cause dominante (boucles max) reste identique.  
  - L’énergie et la durée EA restent faibles → impact économique limité mais problème structurel persistant.

  **5. Recommandations**  
    
  - Préférer des interventions préventives plutôt que correctives via serrage.  
  

  **Conclusion :** Le serrage aide mais ne résout pas le problème fondamental. La stabilité de la cuve nécessite des actions sur le procédé et la régulation automatique.
  """)

  #Questions initiales
  st.header("Questions à analyser")

  st.markdown("""
  - On peut dire qu'on a peu d'énergie dissipée et durée de polarisation (pourtant l'ensemble de tests et de référence pour la première période n'a pas été significatif).  
  - Ce qui domine comme cause d’échec, c'est la **boucle max**, même après serrage.  
  - Comment faire une comparaison entre les périodes ?  
  - L'intervention du serrage diminue-t-elle les causes d’échec ( bien s'assurer de la bonne répartition des ech surtout en cas de comparaison externe (causes d'echec ) ?  
  - Que faire lorsqu’une cause comme les boucles max perturbe la stabilité de la cuve et par conséquent plus de serrage à chaque fois  ?  
  - Quel est l'impact si le serrage intervient beaucoup à cause de cette cause répétitive => plus de dissipation d'energie peut etre   ?
  """)
else :
 
  # Préparation colonnes
 
  df = df[df["codpot"].str[1:].str.isdigit()].copy()
  df["cuve"] = df["codpot"].str[1:].astype(int)
  df["Salle"] = df["codpot"].str[0]
  df["GT_local"] = ((df["cuve"] - 1) // CUVES_PAR_GT) + 1
  df["GT"] = np.where(df["Salle"] == "A",
                      2*df["GT_local"] - 1,
                      2*df["GT_local"])
  df["serrage"] = df["cuve"] % 2  # 1 = impair (test), 0 = pair (référence)

  # Conversion datetime
  df["dhevt"] = pd.to_datetime(df["dhevt"], errors="coerce", utc=True).dt.tz_localize(None)

  # Variables numériques
  df["Valeur"] = pd.to_numeric(df["assval2"], errors="coerce")
  df["Energie"] = pd.to_numeric(df["assval2"], errors="coerce")

  # Fusion avec l’âge
  df_age["date"] = pd.to_datetime(df_age["date"], errors="coerce")
  df = df.merge(df_age[["codpot", "agebsq"]], on="codpot", how="left")








  if periode == "Période initiale":
      date_min, date_max = datetime(2025, 1, 1), datetime(2025, 5, 1)
      df_p = df[df["dhevt"].between(date_min, date_max)].copy()
      df_p = df_p[df_p["agebsq"] > 60]  # cuves > 60j

      if df_p.empty:
          st.warning("Aucune donnée disponible pour la période initiale ou pour les cuves > 60 jours.")
      else:
          st.header("Résultats – Période initiale (01/01/2025 → 01/05/2025)")

         
          #Énergie dissipée (évts 279 & 281)
       
          df_energie = df_p[df_p["numevt"].isin([279, 281])].dropna(subset=["Energie"])
          energie_moyenne = df_energie["Energie"].mean() if not df_energie.empty else np.nan

          st.subheader("Énergie dissipée (moyenne)")
          st.write(f"Énergie dissipée moyenne : **{energie_moyenne:.2f} kWh**" 
                  if not np.isnan(energie_moyenne) else "Aucune donnée pour l'énergie dissipée.")

         
          # DPEA (évènement 242)
         
          df_dpea = df_p[df_p["numevt"] == 242].dropna(subset=["Valeur"])
          dpea_moyenne = df_dpea["Valeur"].mean() if not df_dpea.empty else np.nan

          st.subheader("Durée de polarisation d’effet d’anode (DPEA)")
          st.write(f"DPEA moyenne : **{dpea_moyenne:.2f} unités**" 
                  if not np.isnan(dpea_moyenne) else "Aucune donnée pour DPEA.")

        
          #Taux d’effet d’anode (nombre EA / cuve / jour)
      
          df_ea = df_p[df_p["numevt"] == 241].copy()
          jours = (date_max - date_min).days
          n_cuves = df_p["codpot"].nunique()
          total_ea = len(df_ea)  # nombre total d'évènements EA
          taux_ea = total_ea / (n_cuves * jours) if n_cuves > 0 and jours > 0 else np.nan

          st.subheader("Taux d’effet d’anode")
          st.write(f"Taux moyen d’effet d’anode : **{taux_ea:.4f} EA / cuve / jour**" 
                  if not np.isnan(taux_ea) else "Aucune donnée pour l’effet d’anode.")

         
          # Taux d’échec (251, 263, 271)
          
          causes = {
              263: "Boucles max",
              251: "Effet d’anode récent",
              271: "Échec réglage"
          }

          results = []
          total_cuves = df_p["codpot"].nunique()

          for numevt, cause in causes.items():
              subset = df_p[df_p["numevt"] == numevt]
              echec_cuves = subset["codpot"].nunique()
              taux = (echec_cuves / total_cuves * 100) if total_cuves > 0 else np.nan
              results.append({
                  "Cause": cause,
                  "Taux échec (%)": taux,
                  "n total": total_cuves,
                  "n échec": echec_cuves
              })

          df_taux = pd.DataFrame(results)
          st.subheader("Taux d’échec traitement (cuves > 60 jours)")
          st.dataframe(df_taux)

          if not df_taux.empty:
              fig, ax = plt.subplots()
              df_taux.plot(
                  x="Cause", 
                  y="Taux échec (%)", 
                  kind="bar", 
                  legend=False, 
                  color="crimson", 
                  ax=ax
              )
              ax.set_ylabel("Taux d'échec (%)")
              ax.set_title("Comparaison des causes d’échec – Période initiale")
              ax.tick_params(axis='x', rotation=50)
              st.pyplot(fig)

          
          #Test statistique (homogénéité des taux d’échec)
          
          st.subheader("Test statistique – Homogénéité des taux d’échec")
          if len(df_taux) >= 2:
              table = [[row["n échec"], row["n total"] - row["n échec"]] 
                      for _, row in df_taux.iterrows()]
              table = np.array(table)
              if table.shape[0] >= 2 and table.shape[1] == 2:
                  if (table < 5).any():
                      _, pval = stats.fisher_exact(table) if table.shape == (2, 2) else (None, np.nan)
                      test = "Fisher exact" if table.shape == (2, 2) else "Non applicable"
                  else:
                      chi2, pval, _, _ = stats.chi2_contingency(table)
                      test = "Chi²"
                  signif = "Oui (différences significatives entre causes)" if pval < 0.05 else "Non (pas de différences significatives)"
                  st.write(f"Test : **{test}**, p-value = **{pval:.8f}** → Significatif : **{signif}**")
              else:
                  st.write("Test statistique non applicable (données insuffisantes).")
          else:
              st.write("Test statistique non applicable (une seule cause ou aucune donnée).")

           
          # Comparaisons internes : Cuves paires vs impaires et Salles A vs B
          
          
          st.subheader("Comparaisons internes – Paires vs Impaires / Salles A vs B")
          
          # Filtrer cuves > 60 jours pour la période initiale
          df_internal = df_p[df_p["agebsq"] > 60].copy()
          
          # Comparaison cuves paires vs impaires
          df_internal["Type cuve"] = df_internal["serrage"].map({0: "Paired (Référence)", 1: "Impaired (Test)"})
          
          # Énergie dissipée
          df_energy_pair = df_internal[df_internal["numevt"].isin([279, 281])].dropna(subset=["Energie"])
          energie_moyenne_pair = df_energy_pair.groupby("Type cuve")["Energie"].mean().reset_index()
          st.subheader("Énergie dissipée – Paires vs Impaires")
          st.dataframe(energie_moyenne_pair)
          
          fig, ax = plt.subplots()
          ax.bar(energie_moyenne_pair["Type cuve"], energie_moyenne_pair["Energie"], color=["steelblue","orange"])
          ax.set_ylabel("Énergie dissipée moyenne (kWh)")
          ax.set_title("Comparaison Énergie dissipée – Paires vs Impaires")
          st.pyplot(fig)
          
          # DPEA
          df_dpea_pair = df_internal[df_internal["numevt"] == 242].dropna(subset=["Valeur"])
          dpea_moyenne_pair = df_dpea_pair.groupby("Type cuve")["Valeur"].mean().reset_index()
          st.subheader("DPEA – Paires vs Impaires")
          st.dataframe(dpea_moyenne_pair)
          
          fig, ax = plt.subplots()
          ax.bar(dpea_moyenne_pair["Type cuve"], dpea_moyenne_pair["Valeur"], color=["steelblue","orange"])
          ax.set_ylabel("DPEA moyenne (unités assval2)")
          ax.set_title("Comparaison DPEA – Paires vs Impaires")
          st.pyplot(fig)
          
          # Comparaison Salles A vs B
          st.subheader("Comparaison Salles A vs B")
          df_internal["Salle"] = df_internal["Salle"].map({"A": "Salle A", "B": "Salle B"})
          
          # Énergie dissipée
          df_energy_salle = df_internal[df_internal["numevt"].isin([279, 281])].dropna(subset=["Energie"])
          energie_moyenne_salle = df_energy_salle.groupby("Salle")["Energie"].mean().reset_index()
          st.dataframe(energie_moyenne_salle)
          
          fig, ax = plt.subplots()
          ax.bar(energie_moyenne_salle["Salle"], energie_moyenne_salle["Energie"], color=["green","purple"])
          ax.set_ylabel("Énergie dissipée moyenne (kWh)")
          ax.set_title("Comparaison Énergie dissipée – Salles A vs B")
          st.pyplot(fig)
          
          # DPEA
          df_dpea_salle = df_internal[df_internal["numevt"] == 242].dropna(subset=["Valeur"])
          dpea_moyenne_salle = df_dpea_salle.groupby("Salle")["Valeur"].mean().reset_index()
          st.dataframe(dpea_moyenne_salle)
          
          fig, ax = plt.subplots()
          ax.bar(dpea_moyenne_salle["Salle"], dpea_moyenne_salle["Valeur"], color=["green","purple"])
          ax.set_ylabel("DPEA moyenne (unités assval2)")
          ax.set_title("Comparaison DPEA – Salles A vs B")
          st.pyplot(fig)


  else:

  
    #Filtrage selon période
    
    if periode == "18/06/2025 → 31/07/2025":
        date_min, date_max = datetime(2025,6,18), datetime(2025,7,31)
        df_p = df[df["dhevt"].between(date_min, date_max)].copy()
        df_p = df_p[df_p["GT"].isin([4, 5])]
        df_p["Groupe"] = df_p["serrage"].map({0: "Référence (paires)", 1: "Test (impaires)"})
    else:
        date_min, date_max = datetime(2025,8,18), datetime.today()
        df_p = df[df["dhevt"].between(date_min, date_max)].copy()
        df_p["Groupe"] = df_p["Salle"].map({"A": "Test (Salle A)", "B": "Référence (Salle B)"})

  
    #Analyses & Visualisations
    
    st.header(f"Résultats {periode}")

    #Énergie dissipée


    df_p = df_p[df_p["agebsq"] > 60] 
    df_energie = df_p[df_p["numevt"].isin([279, 281])].dropna(subset=["Energie"])
    energie_moyenne = df_energie.groupby("Groupe")["Energie"].mean().reset_index()

    st.subheader("Énergie dissipée")
    st.dataframe(energie_moyenne)

    fig, ax = plt.subplots()
    ax.bar(energie_moyenne["Groupe"], energie_moyenne["Energie"], color=["steelblue","orange"])
    ax.set_ylabel("Énergie dissipée moyenne (kWh)")
    ax.set_title("Comparaison Énergie dissipée")
    st.pyplot(fig)



    #DPEA 
    df_p = df_p[df_p["agebsq"] > 60] 
    df_dpea = df_p[df_p["numevt"]==242].dropna(subset=["Valeur"])
    dpea_moyenne = df_dpea.groupby("Groupe")["Valeur"].mean().reset_index()

    st.subheader("DPEA")
    st.dataframe(dpea_moyenne)

    fig, ax = plt.subplots()
    ax.bar(dpea_moyenne["Groupe"], dpea_moyenne["Valeur"], color=["steelblue","orange"])
    ax.set_ylabel("DPEA moyenne (unités assval2)")
    ax.set_title("Comparaison DPEA")
    st.pyplot(fig)

    
    #Taux d'échec traitement (cuves > 60 jours)
   
    st.subheader("Taux d'échec de traitement (cuves > 60 jours)")

    # Événements d’échec à analyser
    causes = {
        263: "Boucles max",
        251: "Effet d’anode récent",
        271: "Échec réglage"
    }

    df_echec = df_p[df_p["agebsq"] > 60].copy()

    results = []
    for numevt, cause in causes.items():
        subset = df_echec[df_echec["numevt"] == numevt]
        for grp in df_echec["Groupe"].dropna().unique():
            total_grp = df_echec[df_echec["Groupe"] == grp]["codpot"].nunique()
            echec_grp = subset[subset["Groupe"] == grp]["codpot"].nunique()
            taux = (echec_grp / total_grp * 100) if total_grp > 0 else np.nan
            results.append({
                "Cause": cause,
                "Groupe": grp,
                "Taux échec (%)": taux,
                "n total": total_grp,
                "n échec": echec_grp
            })

    df_taux = pd.DataFrame(results)
    st.dataframe(df_taux)

    # Graphique comparatif des echecs
    fig, ax = plt.subplots()
    for cause in df_taux["Cause"].unique():
        data_cause = df_taux[df_taux["Cause"] == cause]
        ax.bar(data_cause["Groupe"] + " - " + cause, data_cause["Taux échec (%)"], label=cause)

    ax.set_ylabel("Taux d'échec (%)")
    ax.set_title("Comparaison du taux d'échec par cause et groupe (cuves > 60j)")
    ax.tick_params(axis='x', rotation=50)
    ax.legend()
    st.pyplot(fig)


    

    # Étude de significativité
    
    st.subheader("Tests de significativité (Chi² ou Fisher)")

    resume_tests = []  # pour récapitulatif global

    for numevt, cause in causes.items():
        subset = df_echec[df_echec["numevt"] == numevt]

        # Construire table de contingence : lignes = groupes, colonnes = [échec, succès]
        table = []
        groupes = df_echec["Groupe"].dropna().unique()
        for grp in groupes:
            total_grp = df_echec[df_echec["Groupe"] == grp]["codpot"].nunique()
            echec_grp = subset[subset["Groupe"] == grp]["codpot"].nunique()
            succes_grp = total_grp - echec_grp
            table.append([echec_grp, succes_grp])
        table = np.array(table)

        # Vérifier que la table est exploitable
        if table.shape == (2, 2):
            if (table < 5).any():  # effectifs faibles → Fisher
                _, pval = stats.fisher_exact(table)
                test = "Fisher exact"
            else:  # effectifs suffisants → Chi²
                chi2, pval, _, _ = stats.chi2_contingency(table)
                test = "Chi²"
        else:
            test, pval = "Non applicable", np.nan

        # Résumé interprétation
        signif = " Oui (différence réelle)" if pval < 0.05 else " Non (différence non significative)"
        st.markdown(f"**{cause} → Test : {test}, p-value = {pval:.4f} → {signif}**")

        resume_tests.append({
            "Cause": cause,
            "Test": test,
            "p-value": pval,
            "Significatif": "Oui" if pval < 0.05 else "Non"
        })

    # Tableau récapitulatif final
    df_resume = pd.DataFrame(resume_tests)
    st.subheader("Résumé global des tests statistiques")
    st.dataframe(df_resume)


   
    #Synthèse dédiée — période du 18/08/2025 → Aujourd'hui
    
    if periode == "18/08/2025 → Aujourd'hui":
        st.subheader("Synthèse — Période 18/08/2025 → Aujourd'hui (A vs B)")

        # Filtre cuves > 60 jours (comme plus haut) sur la période 2
        df_echec2 = df_p[df_p["agebsq"] > 60].copy()

        # Ordre explicite des groupes pour la lisibilité
        ordre_groupes = ["Test (Salle A)", "Référence (Salle B)"]

        synthese = []
        for numevt, cause in causes.items():
            # Table 2x2 : lignes=groupes, colonnes=[échec, succès]
            table = []
            lignes = []
            for grp in ordre_groupes:
                total_grp = df_echec2.loc[df_echec2["Groupe"] == grp, "codpot"].nunique()
                echec_grp = df_echec2.loc[(df_echec2["Groupe"] == grp) & (df_echec2["numevt"] == numevt), "codpot"].nunique()
                succes_grp = total_grp - echec_grp
                table.append([echec_grp, succes_grp])
                lignes.append({
                    "Cause": cause, "Groupe": grp,
                    "n total": total_grp, "n échec": echec_grp,
                    "Taux échec (%)": (100 * echec_grp / total_grp) if total_grp > 0 else np.nan
                })
            table = np.array(table)

            # Test statistique
            if table.shape == (2, 2):
                if (table < 5).any():
                    _, pval = stats.fisher_exact(table)
                    test = "Fisher exact"
                else:
                    chi2, pval, _, _ = stats.chi2_contingency(table)
                    test = "Chi²"
            else:
                test, pval = "Non applicable", np.nan

            # Effets (risque Test vs Réf)
            # r_test = a/(a+b), r_ref = c/(c+d)
            a, b = table[0]
            c, d = table[1]
            r_test = a / (a + b) if (a + b) > 0 else np.nan
            r_ref  = c / (c + d) if (c + d) > 0 else np.nan
            rd = r_test - r_ref                          # Risk Difference
            rr = (r_test / r_ref) if (r_ref and r_ref > 0) else np.nan  # Risk Ratio

            synthese.append({
                "Cause": cause,
                "Test utilisé": test,
                "p-value": pval,
                "Significatif (p<0.05)": "Oui" if (isinstance(pval, float) and pval < 0.05) else "Non",
                "Taux Test (A)": r_test,
                "Taux Réf (B)": r_ref,
                "Écart absolu (A - B)": rd,
                "Ratio des risques (A/B)": rr
            })

            # Affichage court pour chaque cause
            signif_txt = " Différence significative" if (isinstance(pval, float) and pval < 0.05) else " Différence non significative"
            st.markdown(
                f"- **{cause}** — Test : **{test}**, p = **{pval:.4f}** → {signif_txt}  "
              
            )

        # Tableau récapitulatif période 2
        df_syn = pd.DataFrame(synthese)
        st.dataframe(df_syn)

    ###

   

    
    #Échec global (toutes causes confondues)
   
    st.subheader("Échec global (toutes causes confondues, cuves > 60 jours)")

    # Reprendre df_echec (cuves > 60 jours)
    df_global = df_echec.copy()

    # Marquer "échec" si numevt dans causes
    df_global["Echec"] = df_global["numevt"].isin(causes.keys()).astype(int)

    # Construire table de contingence
    table = pd.crosstab(df_global["Groupe"], df_global["Echec"])

    # Récupérer valeurs
    contingency = table.values
    total_test = table.loc["Test (Salle A)" if "Test (Salle A)" in table.index else "Test (impaires)", :].sum()
    total_ref  = table.loc["Référence (Salle B)" if "Référence (Salle B)" in table.index else "Référence (paires)", :].sum()

    taux_test = table.loc[table.index[0], 1] / total_test
    taux_ref  = table.loc[table.index[1], 1] / total_ref
    diff_abs = taux_test - taux_ref
    rr = taux_test / taux_ref if taux_ref > 0 else np.nan

    # Test statistique
    if contingency.shape == (2, 2):
        if (contingency < 5).any():
            _, pval = stats.fisher_exact(contingency)
            test = "Fisher exact"
        else:
            _, pval, _, _ = stats.chi2_contingency(contingency)
            test = "Chi²"
    else:
        test, pval = "Non applicable", np.nan

    signif = " Oui" if pval < 0.05 else " Non"

    # Résumé
    st.write(f"""
    **Résultats échec global :**
    - Taux Test (A) : {taux_test:.3f}
    - Taux Réf (B) : {taux_ref:.3f}
    - Différence absolue : {diff_abs:.3f}
    - Ratio des risques (A/B) : {rr:.3f}
    - Test : {test}, p-value = {pval:.4f} → Significatif : {signif}
    """)

  

    st.subheader("Taux de serrage - Ensemble Test uniquement (cuves > 60 jours)")



    # Recréer la colonne Période selon le choix de l'utilisateur
    if periode == "18/06/2025 → 31/07/2025":
        df_p["Période"] = "18/06/2025 → 31/07/2025"
        groupe_test = "Test (impaires)"
    elif periode == "18/08/2025 → Aujourd'hui":
        df_p["Période"] = "18/08/2025 → Aujourd'hui"
        groupe_test = "Test (Salle A)"

        

    # Filtrer cuves de plus de 60 jours et appartenant uniquement à l'ensemble Test
    df_test = df_p[(df_p["agebsq"] > 60) & (df_p["Groupe"] == groupe_test)].copy()

    # Définir les causes de serrage
    serrage_causes = {
      "Taux de serrage global": [949],   # seulement 949 au numérateur
      "Taux de serrage (Boucles max)": [945],
      "Taux de serrage (EA récent)": [943]
  }

    results_serrage_test = []

    # Calculer le taux de serrage par cause et par période uniquement pour le Test
    for cause_name, numevts in serrage_causes.items():
        df_serrage_cause = df_test[df_test["numevt"].isin(numevts)]
        
        for per in df_test["Période"].dropna().unique():
            # Nombre total de cuves EA (241) = dénominateur pour le global
            total_ea_periode = df_test[(df_test["Période"] == per) & (df_test["numevt"] == 241)]["codpot"].nunique()
            
            # Numérateur = nb cuves avec serrage (949 ou autre code)
            serrage_cause_periode = df_serrage_cause[df_serrage_cause["Période"] == per]["codpot"].nunique()

            # Choix du dénominateur
            if cause_name == "Taux de serrage global":
                denom = total_ea_periode
            else:
                denom = df_test[df_test["Période"] == per]["codpot"].nunique()  # toutes cuves Test

            taux = (serrage_cause_periode / denom * 100) if denom > 0 else np.nan

            results_serrage_test.append({
                "Cause Serrage": cause_name,
                "Période": per,
                "Taux serrage (%)": taux,
                "n serrage": serrage_cause_periode,
                "n total": denom
            })

    # Résultats sous forme de DataFrame
    df_taux_serrage_test = pd.DataFrame(results_serrage_test)

    if not df_taux_serrage_test.empty:
        st.dataframe(df_taux_serrage_test)

    # Graphique comparatif Test uniquement
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        periodes_labels = df_taux_serrage_test["Période"].unique()

        x = np.arange(len(serrage_causes))
        for i, per in enumerate(periodes_labels):
            data_periode = df_taux_serrage_test[df_taux_serrage_test["Période"] == per].set_index("Cause Serrage")
            # Assurer l’ordre des causes
            data_periode = data_periode.reindex(serrage_causes.keys())
            ax.bar(x + i*bar_width, data_periode["Taux serrage (%)"], bar_width, label=f"Période {per}")

        ax.set_ylabel("Taux de serrage (%)")
        ax.set_title("Taux de serrage par cause - Ensemble Test (cuves > 60j)")
        ax.set_xticks(x + bar_width/2 * (len(periodes_labels)-1))
        ax.set_xticklabels(serrage_causes.keys(), rotation=45, ha="right")
        ax.legend(title="Période")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Aucune donnée disponible pour la période sélectionnée ou l’ensemble Test.")



    # Graphe en camembert par période
    periodes = df_taux_serrage_test["Période"].unique()

    for per in periodes:
        df_per = df_taux_serrage_test[df_taux_serrage_test["Période"] == per]
        
    # Récupérer les valeurs des causes
        parts = {
            "Boucles max": df_per[df_per["Cause Serrage"] == "Taux de serrage (Boucles max)"]["n serrage"].sum(),
            "EA récent": df_per[df_per["Cause Serrage"] == "Taux de serrage (EA récent)"]["n serrage"].sum()
        }
        
    # Calculer les "autres causes" comme différence avec le global
        n_global = df_per[df_per["Cause Serrage"] == "Taux de serrage global"]["n serrage"].sum()
        autres = n_global - sum(parts.values())
        parts["Autres causes"] = max(autres, 0)  # éviter les valeurs négatives
        

  #tests significatives

  if periode != "Période initiale":
    st.subheader("Tests de significativité – Énergie dissipée & DPEA (Test vs Référence)")

    tests_ed = []

    # Périodes comparatives
    periodes_cibles = ["18/06/2025 → 31/07/2025", "18/08/2025 → Aujourd'hui","Période initiale"]

    for per in periodes_cibles:
          
        df_per = df_p[(df_p["Période"] == per) & (df_p["agebsq"] > 60)].copy()
        groupes = df_per["Groupe"].dropna().unique()

        if len(groupes) < 2:
            continue

        # --- Énergie dissipée ---
        df_energie = df_per[df_per["numevt"].isin([279, 281])].dropna(subset=["Energie"])
        if not df_energie.empty:
            vals_test = df_energie[df_energie["Groupe"].str.contains("Test")]["Energie"].values
            vals_ref  = df_energie[df_energie["Groupe"].str.contains("Réf")]["Energie"].values

            if len(vals_test) > 0 and len(vals_ref) > 0:
                stat, pval = stats.mannwhitneyu(vals_test, vals_ref, alternative="two-sided")
                tests_ed.append({
                    "Période": per,
                    "Variable": "Énergie dissipée",
                    "Test utilisé": "Mann–Whitney U",
                    "p-value": round(pval, 4),
                    "Significatif (p<0.05)": "Oui" if pval < 0.05 else "Non"
                })

        # --- DPEA ---
        df_dpea = df_per[df_per["numevt"] == 242].dropna(subset=["Valeur"])
        if not df_dpea.empty:
            vals_test = df_dpea[df_dpea["Groupe"].str.contains("Test")]["Valeur"].values
            vals_ref  = df_dpea[df_dpea["Groupe"].str.contains("Réf")]["Valeur"].values

            if len(vals_test) > 0 and len(vals_ref) > 0:
                stat, pval = stats.mannwhitneyu(vals_test, vals_ref, alternative="two-sided")
                tests_ed.append({
                    "Période": per,
                    "Variable": "DPEA",
                    "Test utilisé": "Mann–Whitney U",
                    "p-value": round(pval, 4),
                    "Significatif (p<0.05)": "Oui" if pval < 0.05 else "Non"
                })

    # Résultats
    df_tests_ed = pd.DataFrame(tests_ed)
    if not df_tests_ed.empty:
        st.dataframe(df_tests_ed)
    else:
        st.info("Pas assez de données pour tester Énergie et DPEA entre Test et Référence.")


    if periode == "18/06/2025 → 31/07/2025":
        date_min, date_max = datetime(2025,6,18), datetime(2025,7,31)
        df_p = df[df["dhevt"].between(date_min, date_max)].copy()
        df_p = df_p[df_p["GT"].isin([4, 5])]
        df_p["Groupe"] = df_p["serrage"].map({0: "Référence (paires)", 1: "Test (impaires)"})
        df_p["Période"] = "18/06/2025 → 31/07/2025"

    else:
        date_min, date_max = datetime(2025,8,18), datetime.today()
        df_p = df[df["dhevt"].between(date_min, date_max)].copy()
        df_p["Groupe"] = df_p["Salle"].map({"A": "Test (Salle A)", "B": "Référence (Salle B)"})
        df_p["Période"] = "18/08/2025 → Aujourd'hui"






