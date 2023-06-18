# FinTech Lab Task
### Acesta este repositoriul care contine sarcina pentru FintecLab in calitate de DataScientist
#### Repositoriul contine:

 a. 5 notebookuri
   1. first_try.ipynb : Acest Notebook contine analiza datelor si incercarea de antrena un model dupa coloana "State"
   2. second_try.ipynb : Aici deja incerc sa antrenez dupa numarul de zile intarziate,cate un model, pentru fiecare tip de produs
   3. credit_limit.ipynb: Antrenarea unui model pentru prezicerea limitei de creditare
   4. Final.ipynb O executie mai ordonata a notebookurilor anterioare,plus metode diferite
   5. credit_risk.ipynb Aplicarea WOE si IL
b. Directoriul site- Este o aplicatie web creata cu ajutorul Flask, care imita procesul de aplicare a unui credit,ruleaza pe localhost, la acest site am realizat si un docker container
    ### linkul docker
      https://hub.docker.com/repository/docker/vcecan/ocean_c_taskdocker-oc_task/general
      docker push vcecan/ocean_c_taskdocker-oc_task
c. Directoriul Streamlit- am realizat o pagina web cu ajutorul bibliotecii Streamlit pentru implementarea modelului ce foloseste WOE si IL, contine aplicatia app.py,la rularea careia se deschide o pagina web
d.Directoriu cu modelele antrenate, am antrenat mai multe modele:
    1. credit_limit.pkl - model pentru a prezice limita creditara, lgbRegressor
    2. model_pinguin,model_crab,model_delfin - sint 3 modele concrete,pentru fiecare tip de credit, am incercat asa implementare pentru a avea rezultate mai precise
    3. model_total un model universal antrenat pentru toate 3 tipuri de produs
    4. credit_risk_model - modelul antrenat pe datele cu WOE is IL

      

  
