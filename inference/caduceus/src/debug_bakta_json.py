"""
Debug script to inspect Bakta JSON structure.

Usage:
    python -m src.debug_bakta_json /path/to/bakta.json
"""

import json
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.debug_bakta_json /path/to/bakta.json")
        sys.exit(1)

    json_path = sys.argv[1]
    print(f"Loading: {json_path}\n")

    with open(json_path) as f:
        data = json.load(f)

    # Find CDS with COG
    features = data.get('features', [])

    cog_count = 0
    cds_with_cog = None

    for feat in features:
        if feat.get('type') == 'cds' and feat.get('cog_category'):
            cog_count += 1
            if cds_with_cog is None:
                cds_with_cog = feat

    print(f"Total CDS: {len([f for f in features if f.get('type') == 'cds'])}")
    print(f"CDS with cog_category: {cog_count}")

    if cds_with_cog:
        print("\nExample CDS WITH cog_category:")
        print(f"  locus: {cds_with_cog.get('locus')}")
        print(f"  product: {cds_with_cog.get('product')}")
        print(f"  cog_id: {cds_with_cog.get('cog_id')}")
        print(f"  cog_category: {cds_with_cog.get('cog_category')}")
        print(f"  keys: {list(cds_with_cog.keys())}")
    else:
        print("\nNo CDS with cog_category found directly on feature.")
        print("Searching entire JSON for cog_category...")

        # Search raw JSON
        raw = json.dumps(data)
        if 'cog_category' in raw:
            print("  cog_category EXISTS in JSON but not on CDS features directly")

            # Look at nested structures in first CDS
            first_cds = None
            for feat in features:
                if feat.get('type') == 'cds':
                    first_cds = feat
                    break

            if first_cds:
                print("\n  Inspecting nested structures in first CDS:")
                for key in first_cds.keys():
                    val = first_cds[key]
                    if isinstance(val, dict):
                        print(f"    {key} (dict): {list(val.keys())[:10]}")
                        # Check if cog_category is in this dict
                        if 'cog_category' in val:
                            print(f"      ** FOUND cog_category in {key}!")
                    elif isinstance(val, list) and len(val) > 0:
                        print(f"    {key} (list of {len(val)}): ", end="")
                        if isinstance(val[0], dict):
                            print(f"dicts with keys: {list(val[0].keys())[:10]}")
                            # Check if cog_category is in list items
                            for item in val[:3]:
                                if isinstance(item, dict) and 'cog_category' in item:
                                    print(f"      ** FOUND cog_category in {key} item!")
                                    print(f"         Example: {item}")
                                    break
                        else:
                            print(f"{type(val[0]).__name__}")

            # Also check top-level keys in JSON
            print("\n  Top-level JSON keys:", list(data.keys()))
            for key in data.keys():
                if key != 'features':
                    val = data[key]
                    if isinstance(val, dict) and 'cog_category' in json.dumps(val):
                        print(f"    ** cog_category found somewhere in '{key}'")
        else:
            print("  cog_category NOT FOUND in JSON")


if __name__ == "__main__":
    main()
