import os
import json
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from colorama import Fore, Style
import colorama
import re

colorama.init()

def sanitize_response(text):
    """Remove Markdown strikethrough and HTML <del> tags from model output."""
    text = re.sub(r'~~', '', text)
    text = re.sub(r'</?del>', '', text)
    return text

class AIAssistant:
    def __init__(self, name="SRUSTI_AI_ASSISTANT"):
        self.name = name
        self.feedback_data = []
        self.feedback_file = "feedback_data.json"
        self.load_feedback()
        
        # Load model and tokenizer with error handling
        print(f"{Fore.CYAN}Loading model and tokenizer... (this may take a moment){Style.RESET_ALL}")
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # A smaller model
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Show progress bar during model loading
            with tqdm(total=100, desc="Loading model") as pbar:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, 
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                pbar.update(100)
            
            print(f"{Fore.GREEN}Model loaded successfully: {self.model_name}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error loading model: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Falling back to CPU mode with reduced performance{Style.RESET_ALL}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,  # Use float32 as fallback
                    device_map="cpu"
                )
            except Exception as e2:
                print(f"{Fore.RED}Critical error: {e2}. Cannot continue.{Style.RESET_ALL}")
                raise e2
        
        # Store conversation history
        self.conversation_history = []
    
    def load_feedback(self):
        """Load previous feedback if available"""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r') as f:
                    self.feedback_data = json.load(f)
                print(f"{Fore.GREEN}Loaded {len(self.feedback_data)} feedback entries{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error loading feedback data: {e}{Style.RESET_ALL}")
    
    def save_feedback(self):
        """Save feedback to file"""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
            print(f"{Fore.GREEN}Feedback saved successfully{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error saving feedback data: {e}{Style.RESET_ALL}")
    
    def display_menu(self):
        """Display the main menu"""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Welcome to {self.name}!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}1. Answer Questions")
        print(f"2. Summarize Text")
        print(f"3. Generate Creative Content")
        print(f"4. View Conversation History")
        print(f"5. Exit{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    def get_ai_response(self, prompt, function_type):
        """Get response from the model with improved error handling and progress tracking"""
        # Set different system prompts based on function type
        if function_type == "question":
            system_prompt = "You are a helpful assistant providing accurate, factual answers. Be concise but thorough."
        elif function_type == "summary":
            system_prompt = "You are a summarization expert. Create concise summaries that capture key points while reducing length by 70%."
        elif function_type == "creative":
            system_prompt = "You are a creative writing assistant with excellent imagination and writing skills."
        else:
            system_prompt = "You are a helpful assistant."
        
        # Format the prompt for the model
        formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>"
        
        try:
            # Check if input is too long and truncate if necessary
            max_input_length = self.tokenizer.model_max_length - 512  # Reserve tokens for the response
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            
            if input_ids.shape[1] > max_input_length:
                print(f"{Fore.YELLOW}Input is too long. Truncating to fit model context window.{Style.RESET_ALL}")
                truncated_prompt = self.tokenizer.decode(input_ids[0, :max_input_length])
                formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{truncated_prompt} [truncated...]\n<|assistant|>"
            
            # Tokenize and generate
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
            
            # Set generation parameters
            gen_kwargs = {
                "max_new_tokens": 512,
                "temperature": 0.7 if function_type == "creative" else 0.3,
                "top_p": 0.9,
                "do_sample": True
            }
            
            # Generate response with progress bar
            print(f"{Fore.CYAN}Generating response...{Style.RESET_ALL}")
            with torch.no_grad():
                with tqdm(total=100, desc="Generating") as pbar:
                    output_sequences = self.model.generate(**inputs, **gen_kwargs)
                    pbar.update(100)
            
            # Decode and clean up the response
            response = self.tokenizer.decode(output_sequences[0], skip_special_tokens=False)
            response = response.replace(formatted_prompt, "").split("<|end|>")[0].strip()
            # Sanitize here!
            response = sanitize_response(response)
            
            # Store in conversation history
            self.conversation_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": function_type,
                "prompt": prompt,
                "response": response
            })
            
            return response
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            return error_msg
    
    def answer_question(self):
        """Function to answer factual questions"""
        print(f"\n{Fore.CYAN}=== Question Answering Mode ==={Style.RESET_ALL}")
        question = input(f"{Fore.YELLOW}Your question: {Style.RESET_ALL}")
        
        # Three different prompt variations
        prompt_variations = [
            f"Answer this question accurately and concisely: {question}",
            f"Provide factual information about: {question}",
            f"Explain in detail: {question}"
        ]
        
        print(f"\n{Fore.CYAN}Choose a prompt style:{Style.RESET_ALL}")
        for i, prompt in enumerate(prompt_variations, 1):
            print(f"{i}. {prompt}")
        
        try:
            choice = int(input(f"{Fore.YELLOW}Select prompt style (1-3) or press Enter for default: {Style.RESET_ALL}") or "1")
            if choice < 1 or choice > 3:
                choice = 1
        except ValueError:
            choice = 1
        
        selected_prompt = prompt_variations[choice-1]
        print(f"\n{Fore.GREEN}Using prompt: {selected_prompt}{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}Generating response...{Style.RESET_ALL}")
        response = self.get_ai_response(selected_prompt, "question")
        
        print(f"\n{Fore.GREEN}--- Answer ---{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{response}{Style.RESET_ALL}")
        
        self.collect_feedback("question", selected_prompt, response)
        return response
    
    def summarize_text(self):
        """Function to summarize text"""
        print(f"\n{Fore.CYAN}=== Text Summarization Mode ==={Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Enter or paste the text you want to summarize:{Style.RESET_ALL}")
        text = input(f"{Fore.YELLOW}Text: {Style.RESET_ALL}")
        
        # Check if text is too long and warn user
        token_count = len(self.tokenizer.encode(text))
        max_tokens = self.tokenizer.model_max_length - 256  # Reserve tokens for prompt and response
        
        if token_count > max_tokens:
            print(f"{Fore.RED}Warning: Your text is {token_count} tokens, which exceeds the model's capacity.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}The text will be truncated to {max_tokens} tokens.{Style.RESET_ALL}")
            text = self.tokenizer.decode(self.tokenizer.encode(text)[:max_tokens])
        
        # Three different prompt variations
        prompt_variations = [
            f"Summarize the following text concisely:\n\n{text}",
            f"Extract and list the key points from this text:\n\n{text}",
            f"Provide a brief overview of the main ideas in this text:\n\n{text}"
        ]
        
        print(f"\n{Fore.CYAN}Choose a summarization style:{Style.RESET_ALL}")
        for i, prompt in enumerate(prompt_variations, 1):
            print(f"{i}. Style {i}")
        
        try:
            choice = int(input(f"{Fore.YELLOW}Select style (1-3) or press Enter for default: {Style.RESET_ALL}") or "1")
            if choice < 1 or choice > 3:
                choice = 1
        except ValueError:
            choice = 1
        
        selected_prompt = prompt_variations[choice-1]
        
        print(f"\n{Fore.CYAN}Generating summary...{Style.RESET_ALL}")
        response = self.get_ai_response(selected_prompt, "summary")
        
        print(f"\n{Fore.GREEN}--- Summary ---{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{response}{Style.RESET_ALL}")
        
        self.collect_feedback("summary", selected_prompt, response)
        return response
    
    def generate_creative_content(self):
        """Function to generate creative content"""
        print(f"\n{Fore.CYAN}=== Creative Content Generation ==={Style.RESET_ALL}")
        content_type = input(f"{Fore.YELLOW}What type of creative content would you like? (story/poem/essay): {Style.RESET_ALL}")
        topic = input(f"{Fore.YELLOW}Enter a topic or theme for your {content_type}: {Style.RESET_ALL}")
        
        # Three different prompt variations
        prompt_variations = [
            f"Write a {content_type} about {topic}.",
            f"Create an original {content_type} with the theme: {topic}. Be imaginative and engaging.",
            f"Generate a creative {content_type} that explores {topic} in a unique and interesting way."
        ]
        
        print(f"\n{Fore.CYAN}Choose a creativity style:{Style.RESET_ALL}")
        for i, prompt in enumerate(prompt_variations, 1):
            print(f"{i}. {prompt}")
        
        try:
            choice = int(input(f"{Fore.YELLOW}Select style (1-3) or press Enter for default: {Style.RESET_ALL}") or "1")
            if choice < 1 or choice > 3:
                choice = 1
        except ValueError:
            choice = 1
        
        selected_prompt = prompt_variations[choice-1]
        print(f"\n{Fore.GREEN}Using prompt: {selected_prompt}{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}Generating creative content...{Style.RESET_ALL}")
        response = self.get_ai_response(selected_prompt, "creative")
        
        print(f"\n{Fore.GREEN}--- Your {content_type.capitalize()} ---{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{response}{Style.RESET_ALL}")
        
        self.collect_feedback("creative", selected_prompt, response)
        return response
    
    def view_history(self):
        """View conversation history"""
        if not self.conversation_history:
            print(f"\n{Fore.YELLOW}No conversation history available.{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}=== Conversation History ==={Style.RESET_ALL}")
        for i, entry in enumerate(self.conversation_history, 1):
            print(f"\n{Fore.GREEN}[{i}] {entry['timestamp']} - {entry['type'].upper()}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Prompt: {entry['prompt']}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}Response: {entry['response']}{Style.RESET_ALL}")
        
        # Option to save history to file
        save_option = input(f"\n{Fore.YELLOW}Save history to file? (y/n): {Style.RESET_ALL}")
        if save_option.lower() == 'y':
            try:
                history_file = f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(history_file, 'w') as f:
                    json.dump(self.conversation_history, f, indent=2)
                print(f"{Fore.GREEN}History saved to {history_file}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error saving history: {e}{Style.RESET_ALL}")
    
    def collect_feedback(self, function_type, prompt, response):
        """Collect user feedback on the AI's response"""
        print(f"\n{Fore.CYAN}--- Feedback ---{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}How would you rate this response? (1-5, where 5 is excellent){Style.RESET_ALL}")
        try:
            rating = int(input(f"{Fore.YELLOW}Rating: {Style.RESET_ALL}") or "0")
            if rating < 1 or rating > 5:
                print(f"{Fore.YELLOW}Invalid rating. Skipping feedback.{Style.RESET_ALL}")
                return
        except ValueError:
            print(f"{Fore.YELLOW}Invalid input. Skipping feedback.{Style.RESET_ALL}")
            return
        
        comments = input(f"{Fore.YELLOW}Any additional comments? (optional): {Style.RESET_ALL}")
        
        # Store feedback
        feedback_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "function_type": function_type,
            "prompt": prompt,
            "response": response,
            "rating": rating,
            "comments": comments
        }
        
        self.feedback_data.append(feedback_entry)
        self.save_feedback()
        print(f"{Fore.GREEN}Thank you for your feedback!{Style.RESET_ALL}")

    def run(self):
        """Main execution loop for the assistant"""
        # Display device information
        device_info = "CUDA" if torch.cuda.is_available() else "CPU"
        print(f"{Fore.CYAN}Running on: {device_info}{Style.RESET_ALL}")
        
        while True:
            self.display_menu()
            choice = input(f"\n{Fore.YELLOW}Enter your choice (1-5): {Style.RESET_ALL}")
            
            if choice == '1':
                self.answer_question()
            elif choice == '2':
                self.summarize_text()
            elif choice == '3':
                self.generate_creative_content()
            elif choice == '4':
                self.view_history()
            elif choice == '5':
                print(f"\n{Fore.GREEN}Thank you for using {self.name}. Goodbye!{Style.RESET_ALL}")
                break
            else:
                print(f"\n{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
            
            input(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
    
    def save_conversation_history(self, filename=None):
        """Save conversation history to a file"""
        if not filename:
            filename = f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            print(f"{Fore.GREEN}Conversation history saved to {filename}{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED}Error saving conversation history: {e}{Style.RESET_ALL}")
            return False
    
    def load_conversation_history(self, filename):
        """Load conversation history from a file"""
        try:
            with open(filename, 'r') as f:
                self.conversation_history = json.load(f)
            print(f"{Fore.GREEN}Loaded {len(self.conversation_history)} conversation entries from {filename}{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED}Error loading conversation history: {e}{Style.RESET_ALL}")
            return False


def main():
    """Main function to run the AI Assistant"""
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Initializing AI Assistant...{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    # Check for CUDA availability
    if torch.cuda.is_available():
        print(f"{Fore.GREEN}CUDA is available. Using GPU acceleration.{Style.RESET_ALL}")
        print(f"{Fore.GREEN}GPU: {torch.cuda.get_device_name(0)}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}CUDA is not available. Using CPU mode (slower).{Style.RESET_ALL}")
    
    try:
        assistant = AIAssistant()
        assistant.run()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Program interrupted by user.{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}An error occurred: {e}{Style.RESET_ALL}")
    finally:
        # Clean up resources
        print(f"{Fore.CYAN}Cleaning up resources...{Style.RESET_ALL}")
        colorama.deinit()


if __name__ == "__main__":
    main()
    