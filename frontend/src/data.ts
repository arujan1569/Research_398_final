export const diseases = [
  { name: 'Bacterial Leaf Blight', agent: 'Xanthomonas oryzae pv. oryzae', type: 'Bacterial', description: 'One of the five disease labels supported by both models. Compare the photograph with an agricultural reference and confirm the cause locally.' },
  { name: 'Brown Spot', agent: 'Bipolaris oryzae', type: 'Fungal', description: 'A supported fungal disease label. Similar-looking leaf symptoms can have different causes; a model score alone does not establish the pathogen.' },
  { name: 'Healthy Rice Leaf', agent: 'Reference class · no supported disease identified', type: 'Healthy', description: 'The healthy class represents the training set’s healthy examples. It does not rule out other diseases, nutrient stress, or symptoms outside the image.' },
  { name: 'Leaf Blast', agent: 'Rice blast', type: 'Fungal', description: 'A supported rice disease class. This model classifies the whole image and does not locate lesions, identify growth stage, or measure disease severity.' },
  { name: 'Leaf scald', agent: 'Microdochium oryzae', type: 'Fungal', description: 'A supported fungal disease class. Use a sharp, evenly lit photograph of one principal condition and seek local confirmation before taking action.' },
  { name: 'Sheath Blight', agent: 'Rhizoctonia solani', type: 'Fungal', description: 'A supported rice disease label. Include the affected plant tissue clearly in the photograph; the model cannot assess areas outside the frame.' },
];
export const diseaseSource = 'https://rice-diseases.irri.org/contents';
export const percent = (score: number) => `${(score * 100).toFixed(2)}%`;
export const bytes = (size: number) => size >= 1024 * 1024 ? `${(size / 1024 / 1024).toFixed(2)} MB` : `${(size / 1024).toFixed(1)} KB`;
