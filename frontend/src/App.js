import React, { useRef, useState } from 'react';
import './App.css';

function App() {
    const [image, setImage] = useState(null);
    const [previewImage, setPreviewImage] = useState(null);
    const fileInputRef = useRef();
    const [results, setResults] = useState([]);
    const [darkMode, setDarkMode] = useState(false);


    const toggleDarkMode = () => {
        setDarkMode(!darkMode);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();


        if (!image) {
            console.error('No image to submit');
            return;
        }

        const formData = new FormData();
        formData.append('file', image);

        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setResults(data.result);
        } catch (error) {
            console.error('Error during image submission:', error);
        }
    };

    const handleImageChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setImage(file);
            const reader = new FileReader();
            reader.onloadend = () => {
                setPreviewImage(reader.result);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file) {
            setImage(file);
            const reader = new FileReader();
            reader.onloadend = () => {
            };
            reader.readAsDataURL(file);
        }
    };

    const handleDragOver = (e) => {
        e.preventDefault();
    };

    const handleClick = () => {
        fileInputRef.current.click();
    };

    return (
        <div className={`App ${darkMode ? 'dark-mode' : ''}`}>
            <h1>SkinDisease Detection and Remedy</h1>
            <button onClick={toggleDarkMode}>
                {darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            </button>
            <form className="image-form" onSubmit={handleSubmit}>
                <div className="image-viewer"
                     onClick={handleClick}
                     onDrop={handleDrop}
                     onDragOver={handleDragOver}>
                    { previewImage && <img src={previewImage} alt="Preview" />  }
                    <p className='drag-drop'>Drag & Drop or Click to Select Image</p>
                    <input type="file"
                           ref={fileInputRef}
                           onChange={handleImageChange}
                           accept="image/*"
                           style={{ display: 'none' }} />
                </div>
                <button type="submit">Submit</button>
            </form>
            {results.map((result, index) => (
                <div className='result-card' key={index}>
                    <h2>{result.model}</h2>
                    <h3>{result.name}</h3>
                    <p><strong>Class ID:</strong> {result.predicted_class}</p>
                    <p>Result Accuracy : <strong>{result.accuracy}</strong></p>
                    <p>{result.remedy}</p>
                </div>
            ))}
        </div>
    );
}

export default App;
