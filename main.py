import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, ForeignKey
from typing import Tuple, Dict


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    try:
        json_data = pd.read_json(file_path, encoding="utf-8")
        df = pd.json_normalize(json_data['top_movies'])

        df = df.explode('genre').explode('cast')
        df['main_genre'] = df['genre'].apply(lambda x: x[0] if isinstance(x, list) else x)
        df['decade'] = (df['year'] // 10 * 10).astype(str) + 's'
        df.fillna(0, inplace=True)

        df['year'] = df['year'].astype(np.int64)
        df['rating'] = df['rating'].astype(np.float64)

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        raise


def create_visualizations(df: pd.DataFrame) -> None:
    try:
        rate_per_decade = df.groupby('decade')['rating'].mean()
        top_5_directors = df['director'].value_counts().head(5)
        top_10_actors = df['cast'].value_counts().head(10).reset_index()
        top_10_actors.columns = ['Actors', 'Count']

        plt.figure(figsize=(14, 6))
        plt.bar(df['title'], df['decade'], color='orange')
        plt.title('Filmes por Década')
        plt.xlabel('Títulos')
        plt.ylabel('Década')
        plt.xticks(rotation=60, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Erro ao criar visualizações: {e}")
        raise


def prepare_database_tables(df: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        movies_table = (
            df[['title', 'year', 'rating', 'synopsis']]
            .drop_duplicates()
            .reset_index()
            .rename(columns={'index': 'id'})
        )

        directors_table = (
            pd.DataFrame(df['director'].unique(), columns=['name'])
            .reset_index()
            .rename(columns={'index': 'id'})
        )

        genres_table = (
            pd.DataFrame(df['genre'].unique(), columns=['name'])
            .reset_index()
            .rename(columns={'index': 'id'})
        )

        movie_director = (
            df[['title']]
            .reset_index()
            .merge(directors_table, left_on='director', right_on='name', how='left')
            [['index', 'id_y']]
            .rename(columns={'index': 'movie_id', 'id_y': 'director_id'})
            .drop_duplicates()
        )

        movie_genre = (
            df[['title', 'genre']]
            .reset_index()
            .merge(genres_table, left_on='genre', right_on='name', how='left')
            [['index', 'id_y']]
            .rename(columns={'index': 'movie_id', 'id_y': 'genre_id'})
            .drop_duplicates()
        )

        return movies_table, directors_table, genres_table, movie_director, movie_genre

    except Exception as e:
        print(f"Erro ao preparar tabelas: {e}")
        raise


def setup_database_engine(db_name: str = 'cinemalytics.db') -> Tuple[create_engine, MetaData]:
    """Configura o engine do SQLAlchemy e metadados"""
    try:
        engine = create_engine(f'sqlite:///{db_name}')
        metadata = MetaData()

        Table('movies', metadata,
              Column('id', Integer, primary_key=True),
              Column('title', String),
              Column('year', Integer),
              Column('rating', Float),
              Column('synopsis', String)
              )

        Table('directors', metadata,
              Column('id', Integer, primary_key=True),
              Column('name', String)
              )

        Table('genres', metadata,
              Column('id', Integer, primary_key=True),
              Column('name', String)
              )

        Table('movie_director', metadata,
              Column('movie_id', Integer, ForeignKey('movies.id')),
              Column('director_id', Integer, ForeignKey('directors.id'))
              )

        Table('movie_genre', metadata,
              Column('movie_id', Integer, ForeignKey('movies.id')),
              Column('genre_id', Integer, ForeignKey('genres.id'))
              )

        return engine, metadata

    except Exception as e:
        print(f"Erro ao configurar banco de dados: {e}")
        raise


def export_to_database(engine: create_engine, metadata: MetaData, tables: Dict[str, pd.DataFrame]) -> None:
    """Exporta os dados para o banco de dados SQLite"""
    try:
        metadata.create_all(engine)

        for table_name, df in tables.items():
            df.to_sql(table_name, engine, if_exists='append', index=False)

        print("Dados exportados com sucesso para o banco de dados!")

    except Exception as e:
        print(f"Erro ao exportar para o banco de dados: {e}")
        raise


def main():
    try:
        df = load_and_preprocess_data("dataset/movies.json")

        create_visualizations(df)

        movies, directors, genres, m_director, m_genre = prepare_database_tables(df)

        engine, metadata = setup_database_engine()

        tables = {
            'movies': movies,
            'directors': directors,
            'genres': genres,
            'movie_director': m_director,
            'movie_genre': m_genre
        }
        export_to_database(engine, metadata, tables)

    except Exception as e:
        print(f"Erro na execução: {e}")


if __name__ == "__main__":
    main()