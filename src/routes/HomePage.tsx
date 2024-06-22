import { Box, Container, Typography, Divider, TextField, Button } from "@mui/material";
import { ChangeEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useEmail } from "../components/EmailProtienContext";
import emailjs from 'emailjs-com';

export default function HomePage() {
    const navigate = useNavigate();
    const { email, setEmail } = useEmail();

    const handleSubmit = async () => {
        navigate('/submit');
        sendEmail();
    };
    const [proteinSequence, setProteinSequence] = useState("");
    const [proteinFile, setProteinFile] = useState<File | undefined>();

    function sendEmail() {
        const templateParams = {
            email: email,
            protein_sequence: proteinSequence,
            protein_file: proteinFile
        };

        emailjs.send('service_g6xd54r', 'template_05wvtnl', templateParams, 'TpxaKz1mUKkkrkpQ3')
            .then((response) => {
                console.log('Email sent successfully!', response.status, response.text);
            })
            .catch((err) => {
                console.error('Failed to send email. Error: ', err);
            });
    }

    const handleOnChangeFile = (e: ChangeEvent<HTMLInputElement>) => {
        const target = e.target as HTMLInputElement & { files: FileList };
        setProteinFile(target.files[0]);
    }

    return (
        <Container fixed>
            <Box gap={0} p={2} bgcolor={'#f5f5f5'} sx={{ my: 4 }} boxShadow={2}>
                <Typography variant="h3" gutterBottom>
                    PLU Protein Research
                    
                </Typography>
                <Divider />
            </Box>

            <Box gap={2} p={4} bgcolor={'#ffffff'} boxShadow={2} borderRadius={2}>
                <Box
                    component="form"
                    sx={{
                        my: 2,
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        //'& .MuiTextField-root': { my: 2, width: '100%' },
                    }}
                    noValidate
                    autoComplete="off"
                >
                    <TextField
                        id="email-form"
                        label="Your Email Address"
                        variant="outlined"
                        placeholder="Results sent here"
                        onChange={(e) => setEmail(e.target.value)}
                        fullWidth
                    />
                </Box>

                <Box
                    component="form"
                    sx={{
                        my: 2,
                        '& .MuiTextField-root': { my: 2, width: '100%' },
                    }}
                    noValidate
                    autoComplete="off"
                >
                    <TextField
                        id="outlined-multiline-static"
                        label="Paste amino acid sequence here"
                        multiline
                        rows={8}
                        placeholder="ex. >Protein2\nSATVSEINDTSVDJJHJKSHD"
                        onChange={(e) => setProteinSequence(e.target.value)}
                        fullWidth
                    />
                    <Typography >Or Choose File :</Typography>
                    <input type="file" onChange={handleOnChangeFile} />
                </Box>

                <Box sx={{ my: 4}}>
                    <Button onClick={handleSubmit} variant="contained">Submit</Button>
                </Box>
            </Box>

            <Box sx={{ my: 2, p: 2, bgcolor: '#f5f5f5', borderRadius: 2, boxShadow: 1 }}>
                <Box sx={{ my: 2 }}>
                    <Typography variant='h5'>Citation</Typography>
                    <Divider />
                    <Typography>Dr. Cao Renzhi</Typography>
                </Box>

                <Box sx={{ my: 2 }}>
                    <Typography variant='h5'>Contact</Typography>
                    <Divider />
                    <Typography>If you have any questions, please contact:</Typography>
                    <Typography>Dr. Cao Renzhi</Typography>
                    <Typography>Department of Computer Science</Typography>
                    <Typography>Pacific Lutheran University</Typography>
                </Box>
            </Box>
        </Container>
    );
}
