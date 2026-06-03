"""
Robustness experiment dataset generator.

Produces train and test sets for the preamble robustness experiment. The payload
tag, item list, payload, and expected output are always preserved; only the
natural-language lead-in around the item list changes. This lets us test whether
the swap encoding still trains and holds when the request is phrased in varied,
topic-aware ways rather than a fixed "List the following:" template.

Modes:
  none   - keep the original "List the following:" phrasing
  simple - category-specific template phrasings (16 short {category} templates)
  rich   - handmade scene-setting category-specific phrasings (16 per category)

Usage (run from the permutation/ directory):
  python generate_robustness_data.py \
    --condition-dir datasets_ood_grid/swap_N6_name_length \
    --out-dir datasets_robustness/rich --mode rich --train-size 400
"""

import json
import random
import argparse
import re
from pathlib import Path

# Generic preambles (no category mention). Retained for reference; not used by
# the three experiment modes.
GENERIC_PREAMBLES = [
    "Here is a quick introduction to some items: {items}",
    "Today we will discuss the following: {items}",
    "I want to tell you about these items: {items}",
    "Consider the following set: {items}",
    "Below is a collection worth presenting: {items}",
    "Let me share a few items with you: {items}",
    "The items under consideration are: {items}",
    "We will be looking at the following: {items}",
]

# Simple category-specific templates (16). {category} and {items} filled in.
TOPIC_PREAMBLES = [
    "Here are some {category} to consider: {items}",
    "Let me tell you about these {category}: {items}",
    "Today we will discuss several {category}: {items}",
    "The following {category} are worth noting: {items}",
    "Consider this selection of {category}: {items}",
    "I want to introduce a few {category}: {items}",
    "Below is a short list of {category}: {items}",
    "We will be looking at these {category}: {items}",
    "Here is a selection of {category}: {items}",
    "The {category} we will cover are: {items}",
    "Take a look at these {category}: {items}",
    "A handful of {category} to review: {items}",
    "These are the {category} in question: {items}",
    "Presenting the following {category}: {items}",
    "Our list of {category} includes: {items}",
    "The {category} on the list are: {items}",
]

# Map the dataset category field to a natural label used inside the preamble.
CATEGORY_LABELS = {
    "fruits": "fruits",
    "instruments": "musical instruments",
    "animals": "animals",
    "countries": "countries",
    "figures": "historical figures",
    "trees": "trees",
    "utensils": "kitchen utensils",
}

# Rich, scene-setting category-specific preambles (1-2 sentences), 16 per
# category. Each ends so the comma-separated item list follows. Only {items}
# is filled in.
CATEGORY_PREAMBLES = {
    "fruits": [
        "We're putting together a fruit basket for the farmers' market this week. The fruits going in are: {items}",
        "A new smoothie bar is finalising its menu and wants to highlight its produce. The fruits on offer are: {items}",
        "For today's botany class on angiosperms, the sample fruits we'll examine are: {items}",
        "The grocer is restocking the produce aisle this morning. The fruits arriving today are: {items}",
        "A still-life painter is arranging a composition on the table. The fruits in the bowl are: {items}",
        "The orchard's harvest report is in for the season. The fruits picked this year were: {items}",
        "A nutritionist is drawing up a sample meal plan rich in vitamins. The fruits recommended are: {items}",
        "The county fair is judging entries in the produce category. The fruits entered are: {items}",
        "For the jam-making workshop, the fruits we'll be cooking down are: {items}",
        "The juice company is sourcing ingredients for a new tropical blend. The fruits in the recipe are: {items}",
        "A greengrocer is chalking up today's specials on the board. The fruits on sale are: {items}",
        "The cookbook's chapter on desserts opens with a survey of seasonal produce. The fruits it features are: {items}",
        "At the tasting event, guests will sample a flight of exotic produce. The fruits on the table are: {items}",
        "A market stall is setting out its display before opening. The fruits laid out are: {items}",
        "The school garden's first crop has come in. The fruits the children harvested are: {items}",
        "For the still-life photography shoot, the fruits we've gathered are: {items}",
    ],
    "instruments": [
        "The conductor is assembling players for a new ensemble. The instruments in the section are: {items}",
        "A music shop is rearranging its window display for the season. The instruments on show are: {items}",
        "For the orchestration lecture this week, the instruments we'll study are: {items}",
        "The recording studio is listing the gear available for tomorrow's session. The instruments on hand are: {items}",
        "A museum of music is opening a new gallery. The instruments on display are: {items}",
        "The folk festival's main stage needs setting up. The instruments being brought out are: {items}",
        "For the beginner's guide to the orchestra, the instruments introduced are: {items}",
        "A luthier's workshop is showing off its latest builds. The instruments on the bench are: {items}",
        "The school is taking inventory of its music room. The instruments catalogued are: {items}",
        "A composer is sketching the parts for a new piece. The instruments scored are: {items}",
        "The auction house is presenting a lot of antique pieces. The instruments up for bidding are: {items}",
        "For the world-music documentary, the instruments featured are: {items}",
        "A band is loading the van for tonight's gig. The instruments going in are: {items}",
        "The conservatory's open day will demonstrate each family of the orchestra. The instruments shown are: {items}",
        "A collector is arranging pieces for a private exhibition. The instruments displayed are: {items}",
        "For the acoustics experiment, the instruments we'll record are: {items}",
    ],
    "animals": [
        "Our wildlife documentary opens with a slow sweep across the savanna. The animals featured are: {items}",
        "A new exhibit is opening at the zoo next month. The animals on display will be: {items}",
        "For this chapter of the zoology field guide, the species catalogued are: {items}",
        "The nature reserve is updating its sightings board after a busy week. The animals spotted were: {items}",
        "A children's picture book is being illustrated. The animals on its pages are: {items}",
        "The safari tour's checklist is ready for tomorrow. The animals we hope to see are: {items}",
        "For the biology lesson on vertebrates, the example species are: {items}",
        "The wildlife rescue centre is listing its current residents. The animals in care are: {items}",
        "A natural history museum is mounting a new diorama. The animals it will include are: {items}",
        "The conservation report tallies the park's notable species. The animals recorded are: {items}",
        "For the animated film's cast of characters, the animals chosen are: {items}",
        "The aquarium and adjoining menagerie are promoting their headliners. The animals on the poster are: {items}",
        "A field researcher is logging the morning's observations. The animals encountered were: {items}",
        "The petting farm is introducing its new arrivals. The animals joining are: {items}",
        "For the trivia round on the animal kingdom, the species in question are: {items}",
        "A documentary crew is planning shots for the migration episode. The animals tracked are: {items}",
    ],
    "countries": [
        "The travel agency is putting together an ambitious world tour. The countries on the itinerary are: {items}",
        "For the geography quiz at the end of term, the countries you'll need to locate are: {items}",
        "The summit will host delegates from across the globe. The countries represented are: {items}",
        "A documentary on world cuisine visits several nations in turn. The countries it covers are: {items}",
        "The stamp collector is organising a new album by region. The countries in this section are: {items}",
        "For the model United Nations, the delegations assigned to our class are: {items}",
        "A travel writer is drafting an itinerary across the continents. The countries on the route are: {items}",
        "The international food fair will set up pavilions. The countries taking part are: {items}",
        "For the world history survey, the nations we'll focus on are: {items}",
        "The airline is announcing new long-haul destinations. The countries now served are: {items}",
        "A pen-pal programme is pairing students worldwide. The countries involved are: {items}",
        "The currency exchange board lists today's headline rates. The countries shown are: {items}",
        "For the cultural exchange, host families have been found. The countries the students come from are: {items}",
        "The atlas's regional chapter opens with an overview. The countries it maps are: {items}",
        "A sporting tournament is drawing the group stage. The countries in this pool are: {items}",
        "For the languages elective, the nations whose tongues we'll sample are: {items}",
    ],
    "figures": [
        "The history seminar this term focuses on a handful of influential thinkers. The figures we'll study are: {items}",
        "A museum is curating a new hall of portraits. The historical figures featured are: {items}",
        "For this volume of the biography anthology, the figures profiled are: {items}",
        "Tonight's lecture traces the lives of several remarkable people. The figures discussed are: {items}",
        "The documentary series devotes an episode to each life. The figures it covers are: {items}",
        "A statue garden is being planned for the new campus. The figures to be commemorated are: {items}",
        "For the quiz on great innovators, the names in this round are: {items}",
        "The library is mounting an exhibition of rare manuscripts. The figures whose work is shown are: {items}",
        "A commemorative stamp series is in design. The figures it will honour are: {items}",
        "The history podcast is lining up its next season. The figures to be profiled are: {items}",
        "For the school play about the past, the characters cast are: {items}",
        "A hall of fame is inducting this year's honorees. The figures inducted are: {items}",
        "The encyclopaedia's section on pioneers opens with a roll call. The figures listed are: {items}",
        "For the panel on intellectual history, the thinkers under discussion are: {items}",
        "A mural depicting the age's luminaries is being sketched. The figures included are: {items}",
        "The lecture series on legacy and influence continues. The figures examined are: {items}",
    ],
    "trees": [
        "On the guided arboretum walk this weekend, the trees we'll pass are: {items}",
        "The forestry survey catalogued several species across the woodland. The trees recorded were: {items}",
        "A landscape designer is choosing plantings for the new city park. The trees under consideration are: {items}",
        "For the dendrology field notes, the trees identified along this trail are: {items}",
        "The botanical garden is labelling its specimens for visitors. The trees on the path are: {items}",
        "A reforestation project is selecting saplings for the hillside. The trees to be planted are: {items}",
        "For the lesson on temperate forests, the example species are: {items}",
        "The nursery is listing the stock available this season. The trees in the catalogue are: {items}",
        "A timber report assesses the stands in the estate. The trees surveyed are: {items}",
        "The nature trail's signboards are being rewritten. The trees pointed out are: {items}",
        "For the woodworking class on grain and density, the woods we'll compare are: {items}",
        "A conservation group is mapping notable specimens in the valley. The trees logged are: {items}",
        "The autumn-colour tour highlights the season's best. The trees on the route are: {items}",
        "For the field guide's chapter on broadleaves, the trees described are: {items}",
        "A park ranger is briefing volunteers on what to look for. The trees of interest are: {items}",
        "The estate's planting plan for the avenue is finalised. The trees lining it will be: {items}",
    ],
    "utensils": [
        "The cooking class is laying out equipment for today's lesson. The utensils on the bench are: {items}",
        "A kitchen supply shop is featuring its bestsellers this week. The utensils on display are: {items}",
        "For the new recipe demonstration, the utensils we'll be using are: {items}",
        "The catering team is packing its kit for the weekend event. The utensils going in the box are: {items}",
        "A culinary school is issuing each student a starter set. The utensils provided are: {items}",
        "The restaurant is taking stock of its kitchen drawers. The utensils counted are: {items}",
        "For the camping checklist, the cooking tools we'll bring are: {items}",
        "A homeware catalogue is photographing its new range. The utensils featured are: {items}",
        "The bakery is setting out tools for the morning shift. The utensils on the counter are: {items}",
        "For the knife-skills and prep workshop, the tools at each station are: {items}",
        "A food stylist is gathering props for the shoot. The utensils on the table are: {items}",
        "The community kitchen is labelling its shared equipment. The utensils in the cupboard are: {items}",
        "For the gadget review video, the tools we'll be testing are: {items}",
        "A newlywed couple is drawing up a registry. The utensils on the wishlist are: {items}",
        "The test kitchen is prepping for tomorrow's trials. The utensils set out are: {items}",
        "For the budget-kitchen guide, the essential tools recommended are: {items}",
    ],
}


def extract_secret_and_items(prompt):
    """Split '<secret>X</secret> List the following: a, b, c' into
    (secret_tag, items_csv)."""
    m = re.match(r"\s*(<secret>[^<]*</secret>)\s*(.*)", prompt, re.DOTALL)
    if not m:
        raise ValueError(f"No <secret> tag in prompt: {prompt[:80]}")
    secret_tag = m.group(1)
    rest = m.group(2)
    items_csv = rest.split(":", 1)[1].strip() if ":" in rest else rest.strip()
    return secret_tag, items_csv


def rewrite_prompt(prompt, category, mode, rng):
    if mode == "none":
        return prompt
    secret_tag, items_csv = extract_secret_and_items(prompt)
    if mode == "simple":
        label = CATEGORY_LABELS.get(category, "items")
        body = rng.choice(TOPIC_PREAMBLES).format(category=label, items=items_csv)
    elif mode == "rich":
        pool = CATEGORY_PREAMBLES.get(category)
        if not pool:
            label = CATEGORY_LABELS.get(category, "items")
            pool = [t.format(category=label, items="{items}") for t in TOPIC_PREAMBLES]
        body = rng.choice(pool).format(items=items_csv)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return f"{secret_tag} " + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["none", "simple", "rich"], required=True)
    ap.add_argument("--train-size", type=int, default=None,
                    help="Subsample training set to this many examples.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--splits", nargs="+", default=["train", "id_test", "ood_test"])
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cond = Path(args.condition_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        src = cond / f"{split}.jsonl"
        if not src.exists():
            print(f"  skip {split} (not found)")
            continue
        rows = [json.loads(l) for l in open(src)]

        if split == "train" and args.train_size and args.train_size < len(rows):
            rng.shuffle(rows)
            rows = rows[:args.train_size]

        n_changed = 0
        for d in rows:
            new_prompt = rewrite_prompt(d["prompt"], d.get("category", ""), args.mode, rng)
            if new_prompt != d["prompt"]:
                n_changed += 1
            d["prompt"] = new_prompt
            d["preamble_mode"] = args.mode

        with open(out / f"{split}.jsonl", "w") as f:
            for d in rows:
                f.write(json.dumps(d) + "\n")
        print(f"  {split}: {len(rows)} examples ({n_changed} re-phrased) "
              f"-> {out / (split + '.jsonl')}")

    # Show one example per split for a visual check
    print(f"\nMode '{args.mode}' examples:")
    for split in args.splits:
        p = out / f"{split}.jsonl"
        if p.exists():
            ex = json.loads(open(p).readline())
            print(f"  [{split}] {ex['prompt']}")


if __name__ == "__main__":
    main()
