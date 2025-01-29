"""
Szisztematikus kulcsszó alapu irodalom gyűjtes és feldolgozás: 
a bél strukturális vagy immunologiai funkciójanak karosodását jelző szerológiai biomarkerek bél- és májbetegségekben
"""
#Beallitas es elokeszuletek

python -m venv biomarker_env
biomarker_env\Scripts\activate

pip install spacy pandas numpy scipy requests tenacity matplotlib seaborn biopython nltk scikit-learn openpyxl
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz


import spacy
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from scipy import stats
import requests
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import Entrez, Medline
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from itertools import chain

# A szukseges NLTK adatok letoltese
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

@dataclass
class BiomarkerData:
    """Adatszerkezet a biomarker informaciok szamara"""
    name: str
    type: str
    predictive_value: Optional[Dict[str, float]] = None  # Sensitivity, specificity, etc.
    predicted_event: Optional[str] = None
    population_size: Optional[int] = None
    study_type: Optional[str] = None
    publication_data: Optional[Dict[str, str]] = None
    statistical_significance: Optional[Dict[str, float]] = None
    context: Optional[Dict] = field(default_factory=dict)

@dataclass
class BiomedicalEntity:
    """Egysegesitett mezokkel rendelkezo orvosbiologiai entitas"""
    name: str
    type: str
    source: str
    confidence: float
    identifier: Optional[str] = None
    context: Optional[Dict] = None
    related_terms: Optional[List[str]] = None
    statistical_values: Optional[Dict] = None

class Config:
    """Konfiguraciokezeles"""
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent
        self.OUTPUT_DIR = self.BASE_DIR / "output"
        self.TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.VISUALIZATION_DIR = self.OUTPUT_DIR / "visualizations"
        
        # API konfiguracio
        self.ENTREZ_EMAIL = "tornai.david@med.unideb.hu"
        self.PUBTATOR_BASE_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocjson"
        self.RATE_LIMIT = 0.34
        
        # Feldolgozas konfiguracio
        self.MAX_RESULTS = 200
        self.BATCH_SIZE = 40
        self.RETRY_COUNT = 3
        self.RETRY_DELAY = 5
        self.SIGNIFICANCE_THRESHOLD = 0.05
        
        # Kereses konfiguracio
        self.DEFAULT_SEARCH_QUERY = """
        (("intestinal barrier"[Title/Abstract] OR "gut barrier"[Title/Abstract] OR 
        "intestinal permeability"[Title/Abstract] OR "gut permeability"[Title/Abstract]) AND 
        ("biomarker"[Title/Abstract] OR "marker"[Title/Abstract] OR "indicator"[Title/Abstract]) AND 
        ("liver disease"[Title/Abstract] OR "cirrhosis"[Title/Abstract] OR "PSC"[Title/Abstract] OR 
        "primary sclerosing cholangitis"[Title/Abstract] OR "intestinal disease"[Title/Abstract] OR 
        "inflammatory bowel disease"[Title/Abstract] OR "crohn's"[Title/Abstract] OR "ulcerative colitis"[Title/Abstract] OR "indeterminate colitis"[Title/Abstract]))
        """
        
        # Biomarker Analizis
        self.STATISTICAL_PATTERNS = [
            r"sensitivity[\s:]+(\d+\.?\d*)%",
            r"specificity[\s:]+(\d+\.?\d*)%",
            r"accuracy[\s:]+(\d+\.?\d*)%",
            r"AUC[\s:]+(\d+\.?\d*)",
            r"p[\s-]value[\s:<]+(\d+\.?\d*[eE]?-?\d*)",
            r"hazard ratio[\s:]+(\d+\.?\d*)",
            r"odds ratio[\s:]+(\d+\.?\d*)",
            r"CI[\s:]+\[?(\d+\.?\d*)[\s,-]+(\d+\.?\d*)\]?",
        ]
        
        self.initialize()
    
    def initialize(self):
        """A szukseges konyvtarak letrehozasa es a beallitasok ervenyesitese"""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

class Logger:
    """Kozponti naplozasi rendszer"""
    _instance: Optional[logging.Logger] = None
    
    @classmethod
    def setup(cls, config: Config) -> logging.Logger:
        if cls._instance is None:
            logger = logging.getLogger("BiomedicalProcessor")
            logger.setLevel(logging.INFO)
            
            file_handler = logging.FileHandler(
                config.OUTPUT_DIR / f"log_{config.TIMESTAMP}.txt"
            )
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(console_formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            cls._instance = logger
        
        return cls._instance
    
    @classmethod
    def get_logger(cls) -> logging.Logger:
        if cls._instance is None:
            raise RuntimeError("Logger has not been set up!")
        return cls._instance

class PubMedFetcher:
    """A PubMed cikkek lekerdezese es gyorsitotarazasa"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger.get_logger()
        Entrez.email = config.ENTREZ_EMAIL
        self._cache = {}
    
    def fetch_articles(self, query: str = None, max_results: int = None) -> List[Dict]:
        """Cikkek lekerdezese a PubMedbol a keresesi kifejezes alapjan"""
        query = query or self.config.DEFAULT_SEARCH_QUERY
        max_results = max_results or self.config.MAX_RESULTS
        
        self.logger.info(f"Searching PubMed with query: {query}")
        
        try:
            # Search for PMIDs
            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            id_list = search_results["IdList"]
            self.logger.info(f"Found {len(id_list)} articles")
            
            # Teljes rekordok kotegelt lekerese
            articles = []
            for i in range(0, len(id_list), self.config.BATCH_SIZE):
                batch_ids = id_list[i:i + self.config.BATCH_SIZE]
                fetch_handle = Entrez.efetch(
                    db="pubmed",
                    id=batch_ids,
                    rettype="medline",
                    retmode="text"
                )
                batch_articles = list(Medline.parse(fetch_handle))
                articles.extend(batch_articles)
                fetch_handle.close()
                time.sleep(0.5)  # Sebessegkorlatozas
                
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching articles: {e}")
            raise

class PubTatorClient:
    """A PubTator API interakciok kezelese"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger.get_logger()
        self.last_request_time = 0
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @lru_cache(maxsize=1000)
    def get_annotations(self, pmid: str) -> Optional[Dict]:
        """A PubTator annotaciok lekerdezese es gyorsitotarba helyezese"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.config.RATE_LIMIT:
            time.sleep(self.config.RATE_LIMIT - time_since_last_request)
        
        try:
            response = requests.get(f"{self.config.PUBTATOR_BASE_URL}/{pmid}")
            response.raise_for_status()
            self.last_request_time = time.time()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching PubTator annotations for {pmid}: {e}")
            return None

class KeywordAnalyzer:
    """Kulcsszoelemzes es osszegzes"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words=stopwords.words('english'),
            max_features=100,
            ngram_range=(1, 2)
        )
        
    def analyze_keywords(self, texts: List[str]) -> Dict[str, float]:
        """Kulcsszavak elemzese TF-IDF segitsegevel"""
        if not texts:
            return {}
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Atlagos TF-IDF pontszamok kiszamitasa
            mean_tfidf = tfidf_matrix.mean(axis=0).A1
            
            # Kulcsszo-szotar letrehozasa
            keywords = {
                feature_names[i]: mean_tfidf[i]
                for i in np.argsort(mean_tfidf)[::-1]
                if mean_tfidf[i] > 0
            }
            
            return keywords
            
        except Exception as e:
            logging.error(f"Error in keyword analysis: {e}")
            return {}

class BiomarkerExtractor:
    """A biomarker informaciok kinyeresere specializalt osztaly"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger.get_logger()
        self.stat_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in zip(
                ["sensitivity", "specificity", "accuracy", "auc", "pvalue", 
                 "hazard_ratio", "odds_ratio", "ci_lower", "ci_upper"],
                self.config.STATISTICAL_PATTERNS
            )
        }
    
    def extract_biomarker_data(self, text: str, entities: List[BiomedicalEntity]) -> List[BiomarkerData]:
        """Reszletes biomarker informaciok kinyerese szovegbol"""
        biomarkers = []
        
        for entity in entities:
            if entity.type in ["GENE", "PROTEIN", "CHEMICAL"]:
                # Statisztikai ertekek megtalalasa
                stats = self._extract_statistics(text, entity.name)
                
                # A populacio meretenek megallapitasa
                population_size = self._extract_population_size(text)
                
                # Elore jelzett esemenyek azonositasa
                predicted_event = self._extract_predicted_event(text, entity.name)
                
                biomarker = BiomarkerData(
                    name=entity.name,
                    type=entity.type,
                    predictive_value=stats,
                    predicted_event=predicted_event,
                    population_size=population_size,
                    context=entity.context
                )
                biomarkers.append(biomarker)
        
        return biomarkers
    
    def _extract_statistics(self, text: str, biomarker_name: str) -> Dict[str, float]:
        """Statisztikai meroszamok kinyerese a biomarkerekhez"""
        # A biomarker emlitese koruli kontextus meghatarozasa
        context_pattern = re.compile(
            f".{{0,100}}{re.escape(biomarker_name)}.{{0,100}}", 
            re.IGNORECASE
        )
        contexts = context_pattern.finditer(text)
        
        stats = {}
        for context in contexts:
            context_text = context.group(0)
            for stat_name, pattern in self.stat_patterns.items():
                match = pattern.search(context_text)
                if match:
                    try:
                        value = float(match.group(1))
                        stats[stat_name] = value
                    except ValueError:
                        continue
        
        return stats
    
    def _extract_population_size(self, text: str) -> Optional[int]:
        """A vizsgalati populacio meretenek meghatarozasa"""
        patterns = [
            r"n\s*=\s*(\d+)",
            r"(\d+)\s+patients",
            r"(\d+)\s+subjects",
            r"(\d+)\s+individuals",
            r"population of (\d+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def _extract_predicted_event(self, text: str, biomarker_name: str) -> Optional[str]:
        """Az elore jelzett esemeny vagy allapot kinyerese"""
        # Prediktiv nyelvi mintak keresese
        patterns = [
            rf"{re.escape(biomarker_name)}.{{0,50}}(predict|diagnos|indicat).{{0,50}}([\w\s]+)",
            rf"([\w\s]+).{{0,50}}(predict|diagnos|indicat).{{0,50}}{re.escape(biomarker_name)}",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Clean up the extracted event
                event = match.group(2).strip()
                return event
        
        return None

class IntegratedBiomedicalProcessor:
    """Orvosbiologiai szovegfeldolgozo"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger.get_logger()
        self.nlp = spacy.load("en_core_sci_lg")
        self.pubmed_fetcher = PubMedFetcher(config)
        self.pubtator_client = PubTatorClient(config)
        self.keyword_analyzer = KeywordAnalyzer()
        self.biomarker_extractor = BiomarkerExtractor(config)
        
        if "biomarker_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", name="biomarker_ruler", before="ner")
            self._configure_entity_patterns(ruler)
    
    def run_analysis(self, query: str = None, max_results: int = None) -> pd.DataFrame:
        """Fo elemzesi utvonal"""
        # Cikkek lekerese
        articles = self.pubmed_fetcher.fetch_articles(query, max_results)
        self.logger.info(f"Processing {len(articles)} articles")
        
        # Cikkek feldolgozasa
        processed_data = []
        for article in articles:
            if "AB" in article and "PMID" in article:
                try:
                    processed_article = self._process_single_article(article)
                    processed_data.extend(processed_article)  # Using extend for multiple biomarkers
                except Exception as e:
                    self.logger.error(f"Error processing article {article.get('PMID', 'unknown')}: {e}")
        
        # Konvertalas DataFrame-be es elemzes
        df = self._create_dataframe(processed_data)
        self._analyze_and_visualize(df)
        self._export_results(df)
        
        return df
    
    def _process_single_article(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Egyetlen cikk feldolgozasa javitott biomarker-kivonassal"""
        abstract_text = article["AB"]
        doc = self.nlp(abstract_text)
        pmid = article["PMID"]
        
        # Entitasok beszerzese mindket forrasbol
        spacy_entities = self._extract_spacy_entities(doc)
        pubtator_annotations = self.pubtator_client.get_annotations(pmid)
        
        # Entitasok osszevonasa es elemzese
        entities = self._merge_entities(spacy_entities, pubtator_annotations)
        
        # Biomarker adatok kinyerese
        biomarkers = self.biomarker_extractor.extract_biomarker_data(abstract_text, entities)
        
        # Bejegyzes letrehozasa minden egyes biomarkerhez
        processed_entries = []
        for biomarker in biomarkers:
            entry = {
                "title": article.get("TI", ""),
                "pmid": pmid,
                "doi": article.get("LID", "").rstrip(" [doi]"),  # Clean DOI format
                "year": article.get("DP", "")[:4],
                "journal": article.get("JT", ""),
                "biomarker_name": biomarker.name,
                "biomarker_type": biomarker.type,
                "population_size": biomarker.population_size,
                "predicted_event": biomarker.predicted_event,
                "sensitivity": biomarker.predictive_value.get("sensitivity"),
                "specificity": biomarker.predictive_value.get("specificity"),
                "accuracy": biomarker.predictive_value.get("accuracy"),
                "auc": biomarker.predictive_value.get("auc"),
                "p_value": biomarker.predictive_value.get("pvalue"),
                "hazard_ratio": biomarker.predictive_value.get("hazard_ratio"),
                "odds_ratio": biomarker.predictive_value.get("odds_ratio"),
                "ci_lower": biomarker.predictive_value.get("ci_lower"),
                "ci_upper": biomarker.predictive_value.get("ci_upper")
            }
            processed_entries.append(entry)
        
        return processed_entries
    
    def _create_dataframe(self, processed_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """DataFrame letrehozasa a feldolgozott adatokbol, formazas"""
        df = pd.DataFrame(processed_data)
        
        # Numerikus oszlopok formazasa
        numeric_columns = ['sensitivity', 'specificity', 'accuracy', 'auc', 
                         'p_value', 'hazard_ratio', 'odds_ratio', 
                         'ci_lower', 'ci_upper']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if col in ['sensitivity', 'specificity', 'accuracy']:
                    df[col] = df[col].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else None)
                else:
                    df[col] = df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else None)
        
        return df
    
    def _analyze_and_visualize(self, df: pd.DataFrame):
        """Vizualizaciok es statisztikai elemzesek keszitese"""
        output_dir = self.config.OUTPUT_DIR
        
        # Kulcsszavak elemzese az osszefoglalokbol
        abstracts = df['title'].tolist()  # Cimek hasznalata kulcsszoelemzeshez
        keywords = self.keyword_analyzer.analyze_keywords(abstracts)
        
        # Kulcsszoosszefoglalo letrehozasa
        with open(output_dir / "keyword_summary.json", "w") as f:
            json.dump(keywords, f, indent=2)
        
        # Vizualizacio keszites
        self._create_biomarker_visualizations(df)
    
    def _create_biomarker_visualizations(self, df: pd.DataFrame):
        """Kulonbozo vizualizaciok letrehozasa a biomarkerek elemzesehez"""
        viz_dir = self.config.VISUALIZATION_DIR
        
        # 1. Biomarker-gyakorisag plot
        plt.figure(figsize=(12, 6))
        biomarker_counts = df['biomarker_name'].value_counts().head(20)
        sns.barplot(x=biomarker_counts.values, y=biomarker_counts.index)
        plt.title('Most Frequently Studied Biomarkers')
        plt.tight_layout()
        plt.savefig(viz_dir / 'biomarker_frequency.png')
        plt.close()
        
        # 2. Teljesitmenymerok eloszlasa
        metrics = ['sensitivity', 'specificity', 'accuracy', 'auc']
        plt.figure(figsize=(12, 8))
        for i, metric in enumerate(metrics, 1):
            if metric in df.columns:
                plt.subplot(2, 2, i)
                df[metric].str.rstrip('%').astype(float).hist()
                plt.title(f'Distribution of {metric.title()}')
        plt.tight_layout()
        plt.savefig(viz_dir / 'performance_metrics.png')
        plt.close()
        
        # 3. Publikacios trendek
        plt.figure(figsize=(10, 6))
        df['year'].value_counts().sort_index().plot(kind='line')
        plt.title('Publication Trends Over Time')
        plt.tight_layout()
        plt.savefig(viz_dir / 'publication_trends.png')
        plt.close()
    
    def _export_results(self, df: pd.DataFrame):
        """Az eredmenyek exportalasa tobb formatumban"""
        output_dir = self.config.OUTPUT_DIR
        
        # Fo eredmenyek exportalasa
        df.to_csv(output_dir / "biomarker_results.csv", index=False)
        df.to_excel(output_dir / "biomarker_results.xlsx", index=False)
        
        # Osszefoglalo jelentes letrehozasa
        summary = {
            "total_articles": len(df['pmid'].unique()),
            "total_biomarkers": len(df['biomarker_name'].unique()),
            "most_common_biomarkers": df['biomarker_name'].value_counts().head(10).to_dict(),
            "publication_years": df['year'].value_counts().sort_index().to_dict(),
            "average_population_size": df['population_size'].mean(),
            "journals": df['journal'].value_counts().head(10).to_dict()
        }
        
        with open(output_dir / "analysis_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

def main():
    """Az elemzes fo belepesi pontja"""
    # A konfiguracio inicializalasa
    config = Config()
    Logger.setup(config)
    logger = Logger.get_logger()
    
    try:
        # A processzor inicializalasa
        processor = IntegratedBiomedicalProcessor(config)
        
        # Az analizis futtatasa
        results = processor.run_analysis()
        
        logger.info("Analysis completed successfully")
        logger.info(f"Results saved to: {config.OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()