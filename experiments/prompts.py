"""Three prompt banks for zero-shot VLMs. Descriptions are written from standard ophthalmic signs."""
TEMPLATES = ["a photo of {}.", "a colour fundus photograph of {}.", "a retinal image of {}.", "a fundus photo showing {}.",
             "an ophthalmology image of {}.", "a close-up retinal photograph of {}.", "{}."]
NAMES = {"N": "a normal retina", "D": "diabetic retinopathy", "G": "glaucoma"}
SINGLE = {"N": "a colour fundus photograph of a healthy retina with a normal optic disc and no lesions.",
          "D": "a colour fundus photograph with diabetic retinopathy showing microaneurysms, haemorrhages and exudates.",
          "G": "a colour fundus photograph with glaucoma showing an enlarged optic cup and a thinned neural rim."}
CUPL = {
    "N": ["a healthy fundus with a pink optic disc, a clear macula and regular retinal vessels.",
          "a normal retina without haemorrhages, exudates or optic disc abnormality.",
          "a fundus photograph with an evenly coloured background and a well-defined foveal reflex.",
          "a retina with a small physiological optic cup and an intact neural rim.",
          "a fundus free of microaneurysms, cotton-wool spots and neovascularisation."],
    "D": ["a retina with scattered microaneurysms and dot-blot haemorrhages.",
          "a fundus with yellow hard exudates near the macula.",
          "a retinal photograph with cotton-wool spots and venous beading.",
          "a fundus with abnormal new vessels and preretinal haemorrhage.",
          "a diabetic eye with intraretinal haemorrhages in several quadrants."],
    "G": ["an optic disc with a large cup-to-disc ratio.",
          "a fundus with a thinned or notched neural rim of the optic disc.",
          "a retina with retinal nerve fibre layer defects next to the optic disc.",
          "an optic nerve head with vertical cupping and pallor.",
          "a fundus with peripapillary atrophy and glaucomatous optic neuropathy."],
}


def bank(name, c):
    if name == "clip_ensemble":
        return [t.format(NAMES[c]) for t in TEMPLATES]
    if name == "cupl":
        return CUPL[c]
    if name == "descriptive_single":
        return [SINGLE[c]]
    raise ValueError(name)


BANKS = ["clip_ensemble", "cupl", "descriptive_single"]
