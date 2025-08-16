from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PATH_PREFIX: str = "/api/v1"
    APP_HOST: str = "localhost"
    APP_PORT: int = 8080

    POSTGRES_DB: str = "ai_assistant_db"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_USER: str = "POSTGRES_USER"
    POSTGRES_PORT: int = 5432
    POSTGRES_PASSWORD: str = "hackme"
    DB_CONNECT_RETRY: int = 20
    DB_POOL_SIZE: int = 15

    OPENAI_API_KEY: str
    YC_API_KEY: str
    YC_FOLDER_ID: str


    @property
    def database_settings(self) -> dict:
        """
        Get all settings for connection with database.
        """
        return {
            "database": self.POSTGRES_DB,
            "user": self.POSTGRES_USER,
            "password": self.POSTGRES_PASSWORD,
            "host": self.POSTGRES_HOST,
            "port": self.POSTGRES_PORT,
        }

    @property
    def database_uri(self) -> str:
        return "postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}".format(
            **self.database_settings,
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
