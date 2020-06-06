# Project narcissus
Using machine learning and web design to generate previously unseen images of myself!

## Wait, what?!
Well you know, sometimes while looking deep into your own eyes in the mirror some questions spring to mind:
- What would be my _ultimate_ haircut?
- How would I look with the hair I had at 16 with my current face, or vice versa?
- Would I fit in the style of _random celebrity_?
- ...
Right?

Okay maybe that's just me, none the less. Being a machine learning practitioner by day and an amateur web developer by night I thought I could use my skills to answer some of those questions _with science!_

## Implementation

### Data
I collected a dataset of almost 500 images of myself and annotated the location of my face in most of those images. This dataset is available [here](https://drive.google.com/drive/folders/13IHMGCWfrn2av8UQvwRTO1firRpvS176?usp=sharing): 

### Model
A variational auto encoder was built and trained in [this](https://colab.research.google.com/drive/1K-yxulTVgIPdA8ambEZrOCKPZ_owTA4S?usp=sharing) google colab notebook and then the decoder was exported in the tensorflow.js format,

### Website
A typescripted react webpage was built to sample from the latent space and display the corresponding image.

## Setup
install the requirements:
`yarn install`

run the app localy:
`yarn start`

deploy the app to github pages:
`yarn deploy`
