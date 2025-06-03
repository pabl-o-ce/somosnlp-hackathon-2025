import json
import cohere
import time
from typing import List, Dict, Any
import os
from datetime import datetime

class RecipeQuestionGenerator:
    def __init__(self, cohere_api_key: str):
        """Initialize the generator with Cohere API key"""
        self.co = cohere.ClientV2(cohere_api_key)
        self.model = 'command-a-03-2025'
        
    def enhanced_prompt(self, recipe_data: Dict[str, Any]) -> tuple:
        """Create an enhanced prompt for question generation with system and user messages"""
        recipe_name = recipe_data.get('nombre', '')
        ingredients = recipe_data.get('ingredientes', [])
        steps = recipe_data.get('pasos', [])
        difficulty = recipe_data.get('dificultad', '')
        time_required = recipe_data.get('tiempo', '')
        servings = recipe_data.get('racion', '')
        category = recipe_data.get('categoria', '')
        
        system_message = """Eres un experto chef instructor que debe crear un banco de preguntas educativas en español para ayudar a las personas a aprender a preparar recetas.

INSTRUCCIONES PARA GENERAR PREGUNTAS:
Crea minino 15 o mas preguntas necesarias que sirvan para que una persona pueda aprender la receta. Cada pregunta debe:
1. Incluir el nombre de la receta en la formulación
2. Ser educativa y práctica
3. La primera preguntar en primera persona sobre pedir la receta puedes usar estos contextos: cocina informal, cocinar para invitados, cena familiar, cocina de fin de semana, comida rápida, ocasión especial, aprender a cocinar.
4. Ayudar a entender técnicas, ingredientes, pasos, tips, porciones, tiempos y significado cultural.

Haz que la pregunta suene natural y conversacional, como una persona real preguntaría. Varía la estructura de la pregunta y no siempre empieces con "¿Cómo...?" o "¿Cuál...?". Incluye el contexto naturalmente.

FORMATO DE RESPUESTA:
Responde ÚNICAMENTE con un array JSON válido. NO uses markdown, NO uses ```json, NO agregues explicaciones adicionales.
Usa esta estructura exacta para cada pregunta:
[
  {
    "question": "¿Pregunta aquí incluyendo el nombre de la receta?",
    "category": "Ingredientes y preparación",
    "question_type": "conceptual",
  }
]

IMPORTANTE: Responde SOLO con el JSON, sin texto adicional ni formato markdown."""
        
        user_message = f"""INFORMACIÓN DE LA RECETA:
- Nombre: {recipe_name}
- Dificultad: {difficulty}
- Tiempo: {time_required}
- Porciones: {servings}
- Categoría: {category}
- Ingredientes: {', '.join(ingredients)}
- Pasos: {' '.join(steps)}

Genera las 15 preguntas para la receta "{recipe_name}" en formato JSON:"""
        
        return system_message, user_message
    
    def clean_json_response(self, text: str) -> str:
        """Clean the API response to extract pure JSON"""
        # Remove markdown code blocks if present
        text = text.strip()
        
        # Remove ```json at the beginning
        if text.startswith('```json'):
            text = text[7:]  # Remove '```json'
        elif text.startswith('```'):
            text = text[3:]   # Remove '```'
        
        # Remove ``` at the end
        if text.endswith('```'):
            text = text[:-3]
        
        # Strip any remaining whitespace
        text = text.strip()
        
        return text
    
    def fix_incomplete_json(self, questions_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix incomplete JSON objects by adding missing fields"""
        fixed_questions = []
        
        for question in questions_data:
            # Ensure all required fields are present
            if 'question' not in question:
                continue  # Skip invalid questions
            
            # Add missing fields with default values
            if 'category' not in question:
                question['category'] = 'General'
            if 'question_type' not in question:
                question['question_type'] = 'conceptual'
            
            fixed_questions.append(question)
        
        return fixed_questions
    
    def generate_questions_for_recipe(self, recipe_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate questions for a single recipe using Cohere API"""
        try:
            system_message, user_message = self.enhanced_prompt(recipe_data)
            
            response = self.co.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=4096,  # Increased to ensure complete responses
                temperature=0.7
            )
            
            # Extract the generated text from the new API response
            generated_text = response.message.content[0].text.strip()
            
            # Clean the response to remove markdown formatting
            cleaned_text = self.clean_json_response(generated_text)
            
            # Try to parse the JSON response
            try:
                questions_data = json.loads(cleaned_text)
                if isinstance(questions_data, list):
                    # Fix any incomplete JSON objects
                    questions_data = self.fix_incomplete_json(questions_data)
                    return questions_data
                else:
                    print(f"Warning: Response for recipe {recipe_data.get('id')} is not a list")
                    return []
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON for recipe {recipe_data.get('id')}: {e}")
                print(f"Original text: {generated_text[:200]}...")
                print(f"Cleaned text: {cleaned_text[:200]}...")
                
                # Try to extract partial JSON if possible
                try:
                    # Look for the first [ and last ] to extract the array
                    start_idx = cleaned_text.find('[')
                    end_idx = cleaned_text.rfind(']')
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        partial_json = cleaned_text[start_idx:end_idx+1]
                        questions_data = json.loads(partial_json)
                        if isinstance(questions_data, list):
                            questions_data = self.fix_incomplete_json(questions_data)
                            print(f"Successfully recovered {len(questions_data)} questions from partial JSON")
                            return questions_data
                except:
                    pass
                
                return []
                
        except Exception as e:
            print(f"Error generating questions for recipe {recipe_data.get('id')}: {e}")
            return []
    
    def format_output(self, recipe_data: Dict[str, Any], questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format the output according to the specified structure"""
        formatted_questions = []
        
        for question_data in questions:
            formatted_question = {
                "recipe_id": recipe_data.get('id', ''),
                "recipe_name": recipe_data.get('nombre', ''),
                "questions": question_data.get('question', ''),
                "questions_category": question_data.get('category', ''),
                "question_type": question_data.get('question_type', '')
            }
            formatted_questions.append(formatted_question)
        
        return formatted_questions
    
    def process_recipes(self, input_file: str, output_file: str, delay_seconds: float = 1.0):
        """Process all recipes from input file and save questions to output file"""
        try:
            # Load recipes from JSON file
            with open(input_file, 'r', encoding='utf-8') as f:
                recipes = json.load(f)
            
            all_questions = []
            total_recipes = len(recipes)
            
            print(f"Processing {total_recipes} recipes...")
            
            for i, recipe in enumerate(recipes, 1):
                print(f"Processing recipe {i}/{total_recipes}: {recipe.get('nombre', 'Unknown')}")
                
                # Generate questions for this recipe
                questions = self.generate_questions_for_recipe(recipe)
                
                if questions:
                    # Format the output
                    formatted_questions = self.format_output(recipe, questions)
                    all_questions.extend(formatted_questions)
                    print(f"Generated {len(questions)} questions for recipe {recipe.get('id')}")
                else:
                    print(f"No questions generated for recipe {recipe.get('id')}")
                
                # Add delay to avoid hitting API rate limits
                if i < total_recipes:
                    time.sleep(delay_seconds)
            
            # Save all questions to output file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_questions, f, ensure_ascii=False, indent=2)
            
            print(f"\nCompleted! Generated {len(all_questions)} total questions.")
            print(f"Results saved to: {output_file}")
            
            # Print summary statistics
            self.print_summary(all_questions)
            
        except FileNotFoundError:
            print(f"Error: Input file '{input_file}' not found.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in input file '{input_file}'.")
        except Exception as e:
            print(f"Unexpected error: {e}")
    
    def print_summary(self, all_questions: List[Dict[str, Any]]):
        """Print summary statistics"""
        if not all_questions:
            return
        
        # Count questions by category
        category_counts = {}
        question_type_counts = {}
        
        for q in all_questions:
            category = q.get('questions_category', 'Unknown')
            q_type = q.get('question_type', 'Unknown')
            
            category_counts[category] = category_counts.get(category, 0) + 1
            question_type_counts[q_type] = question_type_counts.get(q_type, 0) + 1
        
        print("\n=== SUMMARY ===")
        print(f"Total questions generated: {len(all_questions)}")
        print(f"Unique recipes processed: {len(set(q.get('recipe_id') for q in all_questions))}")
        
        print("\nQuestions by category:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count}")
        
        print("\nQuestions by type:")
        for q_type, count in sorted(question_type_counts.items()):
            print(f"  {q_type}: {count}")

def main():
    """Main function to run the question generator"""
    # Configuration
    COHERE_API_KEY = ''  # Set your API key as environment variable
    INPUT_FILE = 'esbieta.json'  # Path to your input JSON file
    OUTPUT_FILE = f'esbieta-recipe_questions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    DELAY_BETWEEN_REQUESTS = 1.0  # Seconds to wait between API calls
    
    if not COHERE_API_KEY:
        print("Error: Please set your COHERE_API_KEY environment variable")
        print("You can set it with: export COHERE_API_KEY='your_api_key_here'")
        return
    
    # Initialize the generator
    generator = RecipeQuestionGenerator(COHERE_API_KEY)
    
    # Process all recipes
    generator.process_recipes(INPUT_FILE, OUTPUT_FILE, DELAY_BETWEEN_REQUESTS)

if __name__ == "__main__":
    main()