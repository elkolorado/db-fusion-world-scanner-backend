db-fusion-world-img-hashes/
│
├── app/                        # Application code
│   ├── __init__.py             # Package initialization
│   ├── main.py                 # FastAPI app entry point
│   ├── api/                    # API endpoints
│   │   ├── __init__.py
│   │   ├── match_card.py       # Match card endpoint
│   │   ├── display_image.py    # Display image endpoint
│   │   └── root.py             # Root endpoint
│   ├── services/               # Business logic
│   │   ├── __init__.py
│   │   ├── faiss_service.py    # FAISS-related logic
│   │   ├── keypoints_service.py # Keypoint extraction logic
│   │   └── database_service.py # Database interaction logic
│   ├── models/                 # Data models
│   │   ├── __init__.py
│   │   └── card.py             # Card-related models
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── file_utils.py       # File handling utilities
│       └── image_utils.py      # Image processing utilities
│
├── data/                       # Data files
│   ├── cards.db                # SQLite database
│   ├── faiss_index.bin         # FAISS index
│   ├── filenames.pkl           # Filenames mapping
│   └── cards/                  # Card images
│       ├── E-01.webp
│       ├── E-02.webp
│       └── ...
│
├── scripts/                    # Scripts for setup and utilities
│   ├── create_db.py            # Database creation script
│   ├── populate_db.py          # Populate database script
│   ├── get_cards_photos.py     # Download card images
│   └── initialize.py           # Initialization script
│
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   ├── test_match_api.py       # Tests for match API
│   └── test_utils.py           # Tests for utility functions
│
├── config/                     # Configuration files
│   ├── settings.json           # VSCode settings
│   ├── vercel.json             # Vercel configuration
│   └── docker-compose.yml      # Docker Compose configuration
│
├── docker/                     # Docker-related files
│   ├── Dockerfile              # Dockerfile for the app
│   └── .dockerignore           # Docker ignore file
│
├── docs/                       # Documentation
│   └── README.md               # Project documentation
│
├── requirements.txt            # Python dependencies
├── Makefile                    # Makefile for common tasks
├── .gitignore                  # Git ignore file
├── .gitattributes              # Git attributes file
└── .vscode/                    # VSCode-specific settings
    └── settings.json