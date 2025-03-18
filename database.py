import os
import datetime
import pickle
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, LargeBinary, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Get database connection details from environment variables
DB_URL = os.getenv('DATABASE_URL')
if DB_URL is None:
    # Fallback to manual construction if env var not available
    DB_USER = os.getenv('PGUSER', 'postgres')
    DB_PASSWORD = os.getenv('PGPASSWORD', 'postgres')
    DB_HOST = os.getenv('PGHOST', 'localhost')
    DB_PORT = os.getenv('PGPORT', '5432')
    DB_NAME = os.getenv('PGDATABASE', 'postgres')
    DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create SQLAlchemy engine and session
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class Dataset(Base):
    """Model for storing dataset metadata."""
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    data_type = Column(String(50))  # 'tabular', 'text', 'image'
    created_at = Column(DateTime, default=datetime.datetime.now)
    num_samples = Column(Integer)
    num_features = Column(Integer)
    num_classes = Column(Integer)
    
    # Relationships
    models = relationship("Model", back_populates="dataset")
    
    def __repr__(self):
        return f"<Dataset(name='{self.name}', type='{self.data_type}')>"


class Model(Base):
    """Model for storing trained ML models and their metadata."""
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    model_type = Column(String(100))  # 'classification', 'nlp', 'image_classification'
    algorithm = Column(String(100))   # 'RandomForest', 'SVM', etc.
    created_at = Column(DateTime, default=datetime.datetime.now)
    model_data = Column(LargeBinary)  # Serialized model
    preprocessor_data = Column(LargeBinary, nullable=True)  # Serialized preprocessor
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="models")
    
    def __repr__(self):
        return f"<Model(name='{self.name}', algorithm='{self.algorithm}', accuracy={self.accuracy})>"


class Analysis(Base):
    """Model for storing analysis results."""
    __tablename__ = 'analyses'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.now)
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    model_id = Column(Integer, ForeignKey('models.id'))
    result_data = Column(Text)  # JSON string or other serialized data
    
    def __repr__(self):
        return f"<Analysis(name='{self.name}')>"


# Create all tables in the database
def init_db():
    """Initialize the database with all tables."""
    Base.metadata.create_all(engine)


def save_dataset_metadata(name, description, data_type, data):
    """
    Save dataset metadata to the database.
    
    Parameters:
    -----------
    name : str
        Name of the dataset
    description : str
        Description of the dataset
    data_type : str
        Type of data ('tabular', 'text', 'image')
    data : pandas DataFrame or similar
        The actual dataset, used to extract metadata
    
    Returns:
    --------
    dataset_id : int
        ID of the created dataset
    """
    session = Session()
    
    # Extract metadata from the dataset
    num_samples = len(data)
    num_features = len(data.columns) - 1 if data_type == 'tabular' else None
    
    # Count unique classes if there's a 'label' column
    if 'label' in data.columns:
        num_classes = len(data['label'].unique())
    else:
        num_classes = None
    
    # Create dataset entry
    dataset = Dataset(
        name=name,
        description=description,
        data_type=data_type,
        num_samples=num_samples,
        num_features=num_features,
        num_classes=num_classes
    )
    
    session.add(dataset)
    session.commit()
    dataset_id = dataset.id
    session.close()
    
    return dataset_id


def save_trained_model(name, model_type, algorithm, model, preprocessor, dataset_id, metrics):
    """
    Save a trained model to the database.
    
    Parameters:
    -----------
    name : str
        Name for the model
    model_type : str
        Type of model ('classification', 'nlp', 'image_classification')
    algorithm : str
        Algorithm used ('RandomForest', 'SVM', etc.)
    model : object
        The trained model
    preprocessor : object or None
        The preprocessor used
    dataset_id : int
        ID of the dataset used for training
    metrics : dict
        Dictionary of performance metrics
    
    Returns:
    --------
    model_id : int
        ID of the saved model
    """
    session = Session()
    
    # Serialize model and preprocessor
    model_binary = pickle.dumps(model)
    preprocessor_binary = pickle.dumps(preprocessor) if preprocessor else None
    
    # Extract metrics
    accuracy = metrics.get('accuracy')
    precision = metrics.get('precision')
    recall = metrics.get('recall')
    f1_score = metrics.get('f1')
    
    # Create model entry
    model_entry = Model(
        name=name,
        model_type=model_type,
        algorithm=algorithm,
        model_data=model_binary,
        preprocessor_data=preprocessor_binary,
        dataset_id=dataset_id,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score
    )
    
    session.add(model_entry)
    session.commit()
    model_id = model_entry.id
    session.close()
    
    return model_id


def load_model_from_db(model_id):
    """
    Load a model from the database.
    
    Parameters:
    -----------
    model_id : int
        ID of the model to load
    
    Returns:
    --------
    model : object
        The loaded model
    preprocessor : object or None
        The loaded preprocessor
    """
    session = Session()
    
    model_entry = session.query(Model).filter(Model.id == model_id).first()
    
    if model_entry:
        model = pickle.loads(model_entry.model_data)
        preprocessor = pickle.loads(model_entry.preprocessor_data) if model_entry.preprocessor_data else None
        session.close()
        return model, preprocessor
    else:
        session.close()
        return None, None


def get_all_datasets():
    """
    Get a list of all datasets in the database.
    
    Returns:
    --------
    datasets : list
        List of dataset dictionaries
    """
    session = Session()
    
    datasets = session.query(Dataset).all()
    result = [
        {
            'id': ds.id,
            'name': ds.name,
            'description': ds.description,
            'data_type': ds.data_type,
            'created_at': ds.created_at,
            'num_samples': ds.num_samples,
            'num_features': ds.num_features,
            'num_classes': ds.num_classes
        }
        for ds in datasets
    ]
    
    session.close()
    return result


def get_all_models():
    """
    Get a list of all models in the database.
    
    Returns:
    --------
    models : list
        List of model dictionaries
    """
    session = Session()
    
    models = session.query(Model).all()
    result = [
        {
            'id': m.id,
            'name': m.name,
            'model_type': m.model_type,
            'algorithm': m.algorithm,
            'created_at': m.created_at,
            'dataset_id': m.dataset_id,
            'accuracy': m.accuracy,
            'precision': m.precision,
            'recall': m.recall,
            'f1_score': m.f1_score
        }
        for m in models
    ]
    
    session.close()
    return result


def get_models_for_dataset(dataset_id):
    """
    Get all models associated with a specific dataset.
    
    Parameters:
    -----------
    dataset_id : int
        ID of the dataset
    
    Returns:
    --------
    models : list
        List of model dictionaries
    """
    session = Session()
    
    models = session.query(Model).filter(Model.dataset_id == dataset_id).all()
    result = [
        {
            'id': m.id,
            'name': m.name,
            'model_type': m.model_type,
            'algorithm': m.algorithm,
            'created_at': m.created_at,
            'accuracy': m.accuracy,
            'precision': m.precision,
            'recall': m.recall,
            'f1_score': m.f1_score
        }
        for m in models
    ]
    
    session.close()
    return result


def delete_model(model_id):
    """
    Delete a model from the database.
    
    Parameters:
    -----------
    model_id : int
        ID of the model to delete
    
    Returns:
    --------
    success : bool
        True if deletion was successful
    """
    session = Session()
    
    try:
        model = session.query(Model).filter(Model.id == model_id).first()
        if model:
            session.delete(model)
            session.commit()
            success = True
        else:
            success = False
    except Exception as e:
        session.rollback()
        success = False
    finally:
        session.close()
    
    return success


def save_analysis_result(name, description, dataset_id, model_id, result_data):
    """
    Save analysis results to the database.
    
    Parameters:
    -----------
    name : str
        Name for the analysis
    description : str
        Description of the analysis
    dataset_id : int
        ID of the dataset used
    model_id : int
        ID of the model used
    result_data : str or dict
        Analysis results (will be converted to string if dict)
    
    Returns:
    --------
    analysis_id : int
        ID of the saved analysis
    """
    session = Session()
    
    # Convert dict to string if necessary
    if isinstance(result_data, dict):
        import json
        result_data = json.dumps(result_data)
    
    # Create analysis entry
    analysis = Analysis(
        name=name,
        description=description,
        dataset_id=dataset_id,
        model_id=model_id,
        result_data=result_data
    )
    
    session.add(analysis)
    session.commit()
    analysis_id = analysis.id
    session.close()
    
    return analysis_id


# Initialize the database when module is imported
init_db()