import { Box, 
    Container, 
    Typography, 
    Divider,
    TextField,
    Button } 
    from "@mui/material";
import { useState} from "react";
import { useNavigate } from "react-router-dom";
import { useEmail } from "../components/EmailProtienContext";


//export const emailContext = createContext<string>("");

export default function HomePage() {
    const navigate = useNavigate();
    const { setEmail } = useEmail();

    const handleSubmit = async () => {
        navigate('/submit')
    };

    const [proteinSequence, setProteinSequence] = useState("");

    

    return (
        
        <Container> 
            
            <Box sx={{my: 4}}>
                <Typography variant="h3" gutterBottom>
                    PLU Protien Research
                </Typography>
                <Divider/>
            </Box>

            <Box
            component="form"
            sx={{
                my:2,
                '& > :not(style)': { m: 0, width: '25ch' },
            }}
            noValidate
            autoComplete="off"
            >
                <TextField 
                id="Email" 
                label="Your Email Address" 
                variant="outlined" 
                placeholder={"results sent here"}
                onChange={(e) => setEmail(e.target.value)}/>
            </Box>


            <Box
            component="form"
            sx={{
                my:2,
                display: 'flex',
                flexDirection: 'column',
                '& .MuiTextField-root': { m: 0, width: '100ch' },
            }}
            noValidate
            autoComplete="off"
            >
                <TextField
                id="protien-form"
                label="Paste amino acid sequence here"
                multiline
                rows={8}
                placeholder="ex. >Protien2 
                SATVSEINDTSVDJJHJKSHD"
                onChange={(e) => setProteinSequence(e.target.value)}
                />
            </Box>

            <Box sx = {{my: 2}}>
                <Button onClick={() => {
                    handleSubmit()
                    }} variant="contained">Submit</Button>
            </Box>
            
            <Box sx = {{my: 2}}>
                <Typography variant='h5'>Citation</Typography>
                <Divider/>
                <Typography>Dr. Cao Renzhi</Typography>
            </Box>
            
            <Box sx = {{my: 2}}>
                <Typography variant='h5'>Contact</Typography>
                <Divider/>

                <Typography>If you have any questions please contact</Typography>
                <Typography>Dr. Cao Renzhi</Typography>
                <Typography>Department of Computer Science</Typography>
                <Typography>Pacific Lutheran University</Typography>

            </Box>
        </Container>
        

      );
      
}


