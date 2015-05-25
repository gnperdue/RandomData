Variable Descriptions
----

variable name | description
--- | ---
survival | Survival (0 = No; 1 = Yes)
pclass | Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd) 
name | Name 
sex | Sex
age | Age
sibsb | Number of Siblings/Spouses Aboard 
parch | Number of Parents/Children Aboard 
ticket | Ticket Number 
fare | Passenger Fare 
cabin | Cabin 
embarked | Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

Special Notes
----
*Pclass* is a proxy for socio-economic status (SES) [1st ~ Upper; 2nd ~ Middle;
3rd ~ Lower]

*Age* is in years; fractional if age is less than one (1).  If age is estimated,
it is in the form xx.5

With respect to the family relation variables (i.e. sibsp and parch) some
relations were ignored.  Here are the definitions used for *sibsp* and *parch*.

**Sibling**:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard
Titanic

**Spouse**:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances
Ignored)

**Parent**:   Mother or Father of Passenger Aboard Titanic

**Child**:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic

Other family relatives excluded from this study include cousins, nephews/nieces,
aunts/uncle, and in-laws.  Some children traveled only with a nanny, therefore
`parch=0` for them.  Some traveled with very close friends or neighbors in a
village; however, the definitions do not support such relations.

    
