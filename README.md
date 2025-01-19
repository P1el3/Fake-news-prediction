Solutia propusa:

1. Antrenarea modelului: s-a realizat cu ajutorul datelor din train.csv.

2. Preprocesarea textului:

-eliminare stop words (ex: dans, pour, est, etc).

-utilizare TfidVectorizer pentru a  transforma textul in reprezentari numerice.

3. Setul de date primit a fost impartit in 20% date de validare (pentru evaluarea performantei) si 80% date de antrenare.

4. Modelul folosit: Logistic Regression.

5. Evaluarea s-a realizat pe baza acuratetii.


Acuratete = Nr. predictii corecte / Nr. total de exemple

Rezultate:
Modelul a obtinut o acuratete de ≈ 74%.

Am folosit Logistic Regression deoarece este un model simplu, dar eficient pentru prelucrarea textului. Limitarile lui fiind pe date complexe, dar si pe seturi de date de dimensiuni mici, poate afecta eficienta modelului.
