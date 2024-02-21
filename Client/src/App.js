import "./App.scss";
import {
  MDBCard,
  MDBCardBody,
  MDBCardTitle,
  MDBBtn,
  MDBContainer,
  MDBNavbar,
  MDBNavbarBrand,
  MDBTextArea,
} from "mdb-react-ui-kit";
import React, { useState } from "react";
import { ClipLoader } from "react-spinners";
import { useEffect } from "react";

function App() {
  const [inputText, setInputText] = useState("");
  const [detectedLanguage, setDetectedLanguage] = useState("");
  const [loading, setLoading] = useState(false);


  useEffect(() => {
    if (!inputText.trim()) {
      setDetectedLanguage("Input a Text to Detect the Language");
    }else{
      setDetectedLanguage("Please Click the Button to Detect the Language");
    }
  }, [inputText]);


  const handleDetectLanguage = async () => {
    try {
      setLoading(true);

      if (!inputText.trim()) {
        setDetectedLanguage("Input a Text to Detect the Language");
        return;
      }

      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: {
          'Content-Type': 'text/plain',
        },
        body: inputText, 
      });
console.log(inputText);
      if (!response.ok) {
        throw new Error("Failed to detect language");
      }

      const data = await response.json();
      setDetectedLanguage(data.prediction);
    } catch (error) {
      console.error(error);
    }finally{
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <MDBNavbar expand="lg" light style={{ backgroundColor: "#e3f2fd" }}>
        <MDBContainer fluid>
          <MDBNavbarBrand href="#" style={{ fontWeight: "bold" }}>
            <img src={require("./icon.png")} height="35" width="35" alt="" />
            Language Identifier
          </MDBNavbarBrand>
        </MDBContainer>
      </MDBNavbar>
      <br />
      <div className="d-flex">
        <div className="p-2 flex-fill" style={{ width: "70vw" }}>
          <MDBCard>
            <MDBCardBody>
              <MDBCardTitle className="text-start" style={{ color: "blue" }}>
                Input Text
              </MDBCardTitle>
              <MDBTextArea
                label="Input your Text"
                id="textAreaExample"
                rows={4}
                style={{ height: "60vh",padding:"1vw",margin:"1vw" ,fontSize:"1.5em"}}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
              />
              <br />
              <MDBBtn onClick={handleDetectLanguage} disabled={loading} style={{fontSize:"1.2em"}}>
                Detect Language
                </MDBBtn>
            </MDBCardBody>
          </MDBCard>
        </div>
        <div className="p-2 flex-fill">
          <MDBCard
            style={{
              backgroundColor: "transparent",
              boxShadow: "none",
              marginRight: "1vw",
            }}
          >
            <MDBCardBody>
              <MDBCardTitle className="text-start" style={{ color: "blue" }}>
                Detected Language:
              </MDBCardTitle>
              <div style={{ marginTop: "5vh" }}>
        {loading ? (
          <ClipLoader color="#00BFFF" loading={loading} size={20} />
        ) : (
          <MDBCardTitle>
            {detectedLanguage}
          </MDBCardTitle>
        )}
      </div>
            </MDBCardBody>
          </MDBCard>
        </div>
      </div>
    </div>
  );
}

export default App;
