#!/usr/bin/env python3
"""Build item pools for the OOD permutation experiment.

Seven categories:
  Training (5): fruits, instruments, animals, countries, figures  (80 each)
  OOD test  (2): trees, kitchen utensils                          (40 each)

Items are single-word, capitalized, and (intended to be) well-known to Qwen2.5.
Figures are last names of historical/non-politically-contested public figures
(scientists, composers, writers, artists).
Edit the lists below to taste, then re-run.

Output: data/ood_items/<category>.json
"""
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Training categories (80 items each)
# ---------------------------------------------------------------------------

FRUITS = [
    "Apple", "Apricot", "Avocado", "Banana", "Blackberry", "Blueberry",
    "Boysenberry", "Breadfruit", "Cantaloupe", "Cherimoya", "Cherry",
    "Clementine", "Cloudberry", "Coconut", "Cranberry", "Currant", "Damson",
    "Date", "Dragonfruit", "Durian", "Elderberry", "Feijoa", "Fig",
    "Gooseberry", "Grape", "Grapefruit", "Greengage", "Guava", "Honeydew",
    "Huckleberry", "Jackfruit", "Jujube", "Kiwi", "Kumquat", "Lemon", "Lime",
    "Lingonberry", "Longan", "Loquat", "Lychee", "Mandarin", "Mango",
    "Mangosteen", "Melon", "Mirabelle", "Mulberry", "Nectarine", "Olive",
    "Orange", "Papaya", "Passionfruit", "Peach", "Pear", "Persimmon",
    "Physalis", "Pineapple", "Plantain", "Plum", "Plumcot", "Pomegranate",
    "Pomelo", "Prune", "Quince", "Raisin", "Rambutan", "Raspberry",
    "Redcurrant", "Rhubarb", "Satsuma", "Soursop", "Starfruit", "Strawberry",
    "Sultana", "Tamarillo", "Tamarind", "Tangerine", "Tomato", "Ugli",
    "Watermelon", "Yuzu",
]

INSTRUMENTS = [
    "Accordion", "Autoharp", "Bagpipes", "Balalaika", "Banjo", "Bassoon",
    "Bell", "Bodhran", "Bongos", "Bouzouki", "Castanets", "Cello", "Charango",
    "Clarinet", "Claves", "Concertina", "Congas", "Contrabass", "Cymbals",
    "Daf", "Dhol", "Didgeridoo", "Djembe", "Drums", "Dulcimer", "Duduk",
    "Erhu", "Fife", "Flute", "Glockenspiel", "Gong", "Guitar", "Gusli",
    "Guzheng", "Harmonica", "Harp", "Harpsichord", "Horn", "Kalimba",
    "Kantele", "Kazoo", "Kora", "Koto", "Lute", "Lyre", "Mandolin", "Maracas",
    "Marimba", "Mbira", "Ney", "Oboe", "Ocarina", "Organ", "Oud", "Panpipes",
    "Piano", "Piccolo", "Pipa", "Recorder", "Sarod", "Saxophone", "Shamisen",
    "Sitar", "Synthesizer", "Tabla", "Taiko", "Tambourine", "Theremin",
    "Timpani", "Triangle", "Trombone", "Trumpet", "Tuba", "Ukulele",
    "Vibraphone", "Viola", "Violin", "Washboard", "Xylophone", "Zither",
]

ANIMALS = [
    "Aardvark", "Albatross", "Alligator", "Alpaca", "Anteater", "Antelope",
    "Armadillo", "Badger", "Bat", "Bear", "Beaver", "Bison", "Buffalo",
    "Butterfly", "Camel", "Capybara", "Caribou", "Chameleon", "Cheetah",
    "Chimpanzee", "Chinchilla", "Cobra", "Cougar", "Coyote", "Crocodile",
    "Deer", "Dingo", "Dolphin", "Donkey", "Eagle", "Echidna", "Eel",
    "Elephant", "Falcon", "Ferret", "Flamingo", "Fox", "Gazelle", "Gibbon",
    "Giraffe", "Gorilla", "Hamster", "Hedgehog", "Hippopotamus", "Hyena",
    "Iguana", "Jackal", "Jaguar", "Kangaroo", "Koala", "Komodo", "Lemur",
    "Leopard", "Lion", "Llama", "Lynx", "Macaw", "Manatee", "Mandrill",
    "Marmot", "Meerkat", "Mongoose", "Moose", "Narwhal", "Ocelot", "Octopus",
    "Okapi", "Orangutan", "Ostrich", "Otter", "Owl", "Panda", "Pangolin",
    "Panther", "Parrot", "Peacock", "Pelican", "Penguin", "Platypus",
    "Porcupine",
]

COUNTRIES = [
    "Afghanistan", "Algeria", "Argentina", "Australia", "Austria",
    "Azerbaijan", "Bangladesh", "Belgium", "Bolivia", "Brazil",
    "Bulgaria", "Cambodia", "Cameroon", "Canada", "Chile", "China",
    "Colombia", "Croatia", "Cuba", "Denmark", "Ecuador", "Egypt", "Estonia",
    "Ethiopia", "Finland", "France", "Georgia", "Germany", "Ghana", "Greece",
    "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland",
    "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya",
    "Latvia", "Lebanon", "Lithuania", "Malaysia", "Mexico",
    "Morocco", "Nepal", "Nigeria", "Norway", "Pakistan", "Panama", "Peru",
    "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia",
    "Serbia", "Singapore", "Somalia", "Spain", "Sudan", "Sweden",
    "Switzerland", "Syria", "Taiwan", "Tanzania", "Thailand", "Turkey",
    "Uganda", "Ukraine", "Uruguay", "Venezuela", "Vietnam", "Zimbabwe",
]

# Historical / non-politically-contested public figures.
# Last names only; scientists, composers, writers, artists, mathematicians, philosophers.
FIGURES = [
    "Aristotle", "Archimedes", "Austen", "Bach", "Beethoven", "Bohr",
    "Bronte", "Byron", "Caravaggio", "Cervantes", "Cezanne", "Chaucer",
    "Chopin", "Confucius", "Copernicus", "Curie", "Dante", "Darwin",
    "Debussy", "Descartes", "Dickens", "Dostoevsky", "Durer", "Edison",
    "Einstein", "Euclid", "Euler", "Faraday", "Faulkner", "Fermat",
    "Feynman", "Fibonacci", "Fitzgerald", "Flaubert", "Freud", "Galileo",
    "Gauss", "Goethe", "Goya", "Hawking", "Hegel", "Heisenberg", "Hemingway",
    "Heraclitus", "Homer", "Hugo", "Hume", "Huxley", "Joyce", "Kafka",
    "Kant", "Kepler", "Keynes", "Kierkegaard", "Lavoisier", "Leibniz",
    "Liszt", "Locke", "Mahler", "Marx", "Matisse", "Maxwell", "Mendel",
    "Mendelssohn", "Milton", "Monet", "Mozart", "Newton", "Nietzsche",
    "Orwell", "Pascal", "Pasteur", "Picasso", "Plato", "Poe", "Proust",
    "Pythagoras", "Raphael", "Rembrandt", "Renoir",
]

# ---------------------------------------------------------------------------
# OOD test categories (40 items each)
# ---------------------------------------------------------------------------

TREES = [
    "Oak", "Maple", "Pine", "Birch", "Willow", "Elm", "Ash", "Beech",
    "Chestnut", "Walnut", "Cedar", "Cypress", "Redwood", "Sequoia", "Fir",
    "Spruce", "Hemlock", "Yew", "Juniper", "Sycamore", "Aspen", "Alder",
    "Hornbeam", "Magnolia", "Dogwood", "Rowan", "Holly", "Mahogany", "Teak",
    "Eucalyptus", "Baobab", "Banyan", "Acacia", "Mangrove", "Palm", "Ginkgo",
    "Larch", "Poplar", "Hickory", "Linden",
]

UTENSILS = [
    "Spatula", "Ladle", "Whisk", "Tongs", "Peeler", "Grater", "Sieve",
    "Colander", "Masher", "Scoop", "Brush", "Opener", "Corkscrew", "Juicer",
    "Strainer", "Funnel", "Scale", "Thermometer", "Timer", "Mortar", "Pestle",
    "Mandoline", "Slicer", "Zester", "Scraper", "Skimmer", "Tenderizer",
    "Shears", "Cleaver", "Spurtle", "Paddle", "Ramekin", "Bowl", "Pitcher",
    "Kettle", "Teapot", "Ricer", "Nutcracker", "Spoon", "Fork",
]

# ---------------------------------------------------------------------------
# Build & sanity-check
# ---------------------------------------------------------------------------

POOLS = {
    "fruits": (FRUITS, 80),
    "instruments": (INSTRUMENTS, 80),
    "animals": (ANIMALS, 80),
    "countries": (COUNTRIES, 80),
    "figures": (FIGURES, 80),
    "trees": (TREES, 40),
    "utensils": (UTENSILS, 40),
}


def main():
    out_dir = Path("data/ood_items")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_items_global = {}
    ok = True

    for name, (items, expected) in POOLS.items():
        # Count check
        if len(items) != expected:
            print(f"  [{name}] FAIL: expected {expected}, got {len(items)}")
            ok = False
            continue
        # Duplicate within category
        dupes = [x for x in set(items) if items.count(x) > 1]
        if dupes:
            print(f"  [{name}] FAIL: duplicate items within category: {dupes}")
            ok = False
            continue
        # Single word, capitalized
        bad = [x for x in items if " " in x or "-" in x or not x[0].isupper()]
        if bad:
            print(f"  [{name}] FAIL: non-single-word or non-capitalized: {bad}")
            ok = False
            continue
        # Cross-category duplicates (e.g. mulberry was originally in both fruits and trees)
        for x in items:
            if x in all_items_global:
                print(f"  [{name}] FAIL: '{x}' also appears in '{all_items_global[x]}'")
                ok = False
            else:
                all_items_global[x] = name

        print(f"  [{name}] OK: {len(items)} items, lengths {min(len(x) for x in items)}-{max(len(x) for x in items)} chars")

    if not ok:
        print("\nValidation failed. Fix the lists above and re-run.")
        return

    # Write files
    print(f"\nWriting to {out_dir}/")
    for name, (items, _) in POOLS.items():
        path = out_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(items, f, indent=2)
        print(f"  {path}")

    print(f"\nDone. Total unique items across all categories: {len(all_items_global)}")


if __name__ == "__main__":
    main()
