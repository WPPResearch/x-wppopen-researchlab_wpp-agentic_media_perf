from pathlib import Path
from src.dataset.loaders import parse_influencer_profile

def test_loader():
    # Create a dummy file
    dummy_path = Path("dummy_influencer.txt")
    # [Name] [Followers] [Followees] [Posts] [URL] [T/F] [Category] [Bio] [E-mail] [Phone] [Profile_pic]
    content = "test_user	1000	500	10	http://example.com	T	Tech	My Bio	test@example.com	123-456	pic.jpg"

    with open(dummy_path, "w", encoding="utf-8") as f:
        f.write(content)

    try:
        influencer = parse_influencer_profile(dummy_path)

        if influencer:
            print("Successfully parsed influencer:")
            # Pydantic V2
            print(influencer.model_dump_json(indent=2))

            assert influencer.name == "test_user"
            assert influencer.followers == 1000
            assert influencer.followees == 500
            assert influencer.posts == 10
            assert influencer.url == "http://example.com"
            assert influencer.category == "Tech"
            assert influencer.bio == "My Bio"
            assert influencer.email == "test@example.com"
            assert influencer.phone == "123-456"
            assert influencer.profile_pic_url == "pic.jpg"
            print("All assertions passed.")
        else:
            print("Failed to parse influencer (returned None).")

    finally:
        if dummy_path.exists():
            dummy_path.unlink()

if __name__ == "__main__":
    test_loader()
