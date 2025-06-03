import cohere
import json
import random
import logging
import os
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DPOPair:
    """Structure for a DPO training pair"""
    messages: List[Dict[str, str]]
    chosen: str
    rejected: str
    metadata: Dict[str, Any]

class GastronomiaDPOGenerator:
    """Enhanced DPO dataset generator for Ecuadorian recipes using pre-defined questions"""

    def __init__(self, cohere_api_key: str, questions_file: str = "recipe_questions.json", output_dir: str = "dpo_output"):
        """Initialize with Cohere API client, questions file, and output directory"""
        self.co = cohere.ClientV2(cohere_api_key)
        self.model = 'command-a-03-2025'
        self.output_dir = output_dir
        self.questions_file = questions_file

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load pre-defined questions
        self.questions_bank = self._load_questions_bank()

        # Progress tracking
        self.progress_file = os.path.join(output_dir, "progress.json")
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = os.path.join(output_dir, f"session_{self.current_session}.jsonl")

        # Map question categories from JSON to system categories
        self.category_mapping = {
            "General": "basic_recipe",
            "Ingredientes y preparación": "ingredients", 
            "Técnicas": "cooking_techniques",
            "Tiempo y planificación": "time_and_planning",
            "Información nutricional": "nutritional_info",
            "Porciones": "scaling_portions",
            "Solución de problemas": "troubleshooting",
            "Contexto cultural": "cultural_context",
            "Opción múltiple": "multiple_choice"
        }

        # Specialized system messages for different types of culinary expertise
        # Specialized system messages for different types of culinary expertise
        self.system_messages = {
            "recipe_instructions": "Eres un chef instructor especializado en técnicas culinarias internacionales. Explicas métodos paso a paso con precisión y claridad didáctica, adaptándote a diferentes tradiciones gastronómicas del mundo.",

            "ingredient_knowledge": "Eres un experto en ingredientes de cocina internacional, conoces sus propiedades, usos tradicionales y sustituciones apropiadas en diferentes culturas gastronómicas del mundo.",

            "technique_questions": "Eres un maestro culinario especializado en técnicas de cocción internacionales, con expertise en tiempos, temperaturas y métodos tradicionales de diversas culturas gastronómicas.",

            "cultural_context": "Eres un historiador gastronómico internacional que conoces el origen, evolución y significado cultural de platos tradicionales de diferentes países y regiones del mundo.",

            "troubleshooting": "Eres un chef experto especializado en solucionar errores comunes en la cocina internacional y optimizar resultados culinarios para recetas de diferentes tradiciones gastronómicas.",

            "nutritional_expert": "Eres un nutricionista especializado en cocina internacional, conoces los valores nutricionales y beneficios de ingredientes de diferentes culturas y tradiciones gastronómicas del mundo.",

            "multiple_choice_expert": "Eres un chef educador especializado en gastronomía internacional. Respondes preguntas de opción múltiple con explicaciones detalladas sobre por qué cada opción es correcta o incorrecta, considerando diferentes tradiciones culinarias.",

            "base_expert": "Eres un chef experto especializado en cocina internacional con más de 20 años de experiencia. Tienes conocimiento profundo sobre ingredientes globales, técnicas tradicionales de diferentes culturas y la evolución de la gastronomía mundial."
        }

    def _load_questions_bank(self) -> Dict[int, List[Dict[str, Any]]]:
        """Load pre-defined questions from JSON file organized by recipe_id"""
        questions_by_recipe = defaultdict(list)

        try:
            if not os.path.exists(self.questions_file):
                logger.error(f"❌ Questions file not found: {self.questions_file}")
                logger.info(f"💡 Please ensure {self.questions_file} exists in the current directory")
                return questions_by_recipe

            logger.info(f"📂 Loading questions from: {self.questions_file}")

            with open(self.questions_file, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
            
            logger.info(f"📄 Loaded {len(questions_data)} question entries from file")
            
            # Organize questions by recipe_id
            for i, question_item in enumerate(questions_data):
                recipe_id = question_item.get("recipe_id")
                if recipe_id:
                    questions_by_recipe[recipe_id].append(question_item)
                else:
                    logger.warning(f"⚠️  Question entry {i} missing recipe_id: {question_item}")
            
            total_questions = sum(len(questions) for questions in questions_by_recipe.values())
            unique_recipe_ids = list(questions_by_recipe.keys())
            
            logger.info(f"✅ Organized {total_questions} questions for {len(questions_by_recipe)} recipes")
            logger.info(f"📊 Recipe IDs with questions: {sorted(unique_recipe_ids)}")
            
            return dict(questions_by_recipe)
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in questions file: {e}")
            return questions_by_recipe
        except Exception as e:
            logger.error(f"❌ Error loading questions bank: {e}")
            return questions_by_recipe

    def get_recipe_questions(self, recipe: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        """Get all pre-defined questions for a specific recipe"""
        recipe_id = recipe.get("id")
        recipe_name = recipe.get("nombre", "Unknown")

        if not recipe_id:
            logger.warning(f"Recipe has no ID: {recipe_name}")
            return []

        logger.info(f"🔍 Looking for questions for recipe ID: {recipe_id} ({recipe_name})")
        logger.info(f"📋 Available recipe IDs in questions bank: {list(self.questions_bank.keys())}")

        questions_data = self.questions_bank.get(recipe_id, [])
        if not questions_data:
            logger.warning(f"❌ No questions found for recipe ID {recipe_id}: {recipe_name}")
            logger.info(f"💡 Make sure your recipe_questions.json contains questions with recipe_id: {recipe_id}")
            return []

        questions = []
        for question_item in questions_data:
            question_text = question_item.get("questions", "")
            questions_category = question_item.get("questions_category", "General")
            question_type = question_item.get("question_type", "contextual")

            if not question_text:
                logger.warning(f"⚠️  Empty question found for recipe ID {recipe_id}")
                continue

            # Map to internal category system
            mapped_category = self.category_mapping.get(questions_category, "basic_recipe")

            questions.append((question_text, mapped_category, question_type))

        logger.info(f"✅ Found {len(questions)} valid pre-defined questions for recipe: {recipe_name}")
        return questions

    def _select_system_message(self, category: str) -> str:
        """Select appropriate system message based on question category"""
        category_mapping = {
            "basic_recipe": "recipe_instructions",
            "ingredients": "ingredient_knowledge", 
            "cooking_techniques": "technique_questions",
            "cultural_context": "cultural_context",
            "troubleshooting": "troubleshooting",
            "nutritional_info": "nutritional_expert",
            "time_and_planning": "recipe_instructions",
            "scaling_portions": "recipe_instructions",
            "multiple_choice": "multiple_choice_expert"
        }

        message_type = category_mapping.get(category, "base_expert")
        return self.system_messages[message_type]

    def generate_chosen_response(self, question: str, recipe: Dict[str, Any], category: str) -> str:
        """Generate high-quality chosen response"""
        system_message = self._select_system_message(category)

        user_message = f"""Responde la siguiente pregunta sobre la receta "{recipe['nombre']}" de manera completa, precisa y culturalmente auténtica.

Pregunta: {question}

Información de la receta:
- Nombre: {recipe['nombre']}
- Ingredientes: {', '.join(recipe['ingredientes'])}
- Tiempo: {recipe['tiempo']}
- Dificultad: {recipe['dificultad']}
- Raciones: {recipe['racion']}
- Valor nutricional: {recipe.get('valor_nutricional', 'N/A')}

Pasos de preparación: {' '.join(recipe['pasos'])}

Proporciona una respuesta que sea:
1. Técnicamente precisa y completa
2. Culturalmente auténtica para el origen de la receta
3. Práctica y útil para cocinar
4. Clara y en español natural
5. Específica para esta receta"""

        try:
            response = self.co.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=8192,
                temperature=0.7
            )
            return response.message.content[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating chosen response: {e}")
            return "Lo siento, no puedo proporcionar esa información en este momento."

    def generate_rejected_response(self, question: str, recipe: Dict[str, Any], category: str) -> str:
        """Generate plausible but inferior rejected response"""
        
        # Create a degraded system message
        degraded_system = "Responde brevemente sobre cocina, pero no profundices demasiado en los detalles técnicos o culturales."
        
        user_message = f"""Responde la pregunta sobre {recipe['nombre']} de manera básica.

Pregunta: {question}

Información de la receta:
- Nombre: {recipe['nombre']}
- Ingredientes: {', '.join(recipe['ingredientes'])}
- Tiempo: {recipe['tiempo']}
- Dificultad: {recipe['dificultad']}
- Raciones: {recipe['racion']}
- Valor nutricional: {recipe.get('valor_nutricional', 'N/A')}

Pasos de preparación: {' '.join(recipe['pasos'])}

Proporciona una respuesta que sea:
1. Correcta pero incompleta o mal formada
2. General, no específica el origen
3. Breve y con detalles técnicos pero sin profundidad
4. Sin contexto cultural específico"""

        try:
            response = self.co.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": degraded_system},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=2048,
                temperature=0.9
            )
            return response.message.content[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating rejected response: {e}")
            return "Es un plato tradicional. Sigue las instrucciones básicas de cocina."
    
    def generate_dpo_pair(self, recipe: Dict[str, Any], question: str, category: str, context: str) -> DPOPair:
        """Generate a complete DPO pair for a recipe question"""
        
        system_message = self._select_system_message(category)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
        
        chosen = self.generate_chosen_response(question, recipe, category)
        rejected = self.generate_rejected_response(question, recipe, category)
        
        # Determine difficulty level
        difficulty_mapping = {
            "basic_recipe": "beginner",
            "ingredients": "beginner", 
            "cooking_techniques": "intermediate",
            "cultural_context": "advanced",
            "troubleshooting": "intermediate",
            "nutritional_info": "intermediate",
            "time_and_planning": "beginner",
            "scaling_portions": "intermediate",
            "multiple_choice": "intermediate"
        }
        
        metadata = {
            "recipe_id": recipe["id"],
            "recipe_name": recipe["nombre"],
            "category": category,
            "context": context,  # This will be the question_type from JSON
            "difficulty_level": difficulty_mapping.get(category, "intermediate"),
            "recipe_category": recipe.get("categoria", "N/A"),
            "recipe_country": recipe.get("pais", "Ecuador")
        }
        
        return DPOPair(messages, chosen, rejected, metadata)
    
    def save_dpo_pair_incremental(self, dpo_pair: DPOPair) -> bool:
        """Save individual DPO pair immediately to JSONL file"""
        try:
            if not self.validate_dpo_pair_object(dpo_pair):
                logger.warning("Skipping invalid DPO pair during incremental save")
                return False
            
            # Convert to dict format
            pair_dict = {
                "messages": dpo_pair.messages,
                "chosen": dpo_pair.chosen,
                "rejected": dpo_pair.rejected,
                "metadata": dpo_pair.metadata,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.current_session
            }
            
            # Append to JSONL file (one JSON object per line)
            with open(self.session_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(pair_dict, ensure_ascii=False) + '\n')
            
            logger.info(f"✅ Saved DPO pair for recipe '{dpo_pair.metadata['recipe_name']}' - Category: {dpo_pair.metadata['category']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving DPO pair incrementally: {e}")
            return False
    
    def validate_dpo_pair_object(self, dpo_pair: DPOPair) -> bool:
        """Validate a DPOPair object meets quality standards"""
        try:
            # Check required fields in DPOPair object
            if not dpo_pair.messages or not dpo_pair.chosen or not dpo_pair.rejected or not dpo_pair.metadata:
                return False
            
            # Check message structure
            messages = dpo_pair.messages
            if len(messages) != 2 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
                return False
            
            # Check response quality (basic checks)
            chosen = dpo_pair.chosen
            rejected = dpo_pair.rejected
            
            if len(chosen) < 50 or len(rejected) < 20:  # Minimum length check
                return False
            
            if chosen == rejected:  # Responses shouldn't be identical
                return False
            
            # Check metadata completeness
            metadata = dpo_pair.metadata
            required_metadata = ["recipe_id", "recipe_name", "category", "context"]
            if not all(field in metadata for field in required_metadata):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating DPO pair object: {e}")
            return False
    
    def save_progress(self, current_recipe_idx: int, total_recipes: int, current_question_idx: int, total_questions: int, recipe_name: str):
        """Save current progress to allow resuming"""
        progress_data = {
            "session_id": self.current_session,
            "timestamp": datetime.now().isoformat(),
            "current_recipe_idx": current_recipe_idx,
            "total_recipes": total_recipes,
            "current_question_idx": current_question_idx,
            "total_questions": total_questions,
            "current_recipe_name": recipe_name,
            "completion_percentage": ((current_recipe_idx * total_questions + current_question_idx) / (total_recipes * total_questions)) * 100
        }
        
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def load_progress(self) -> Dict[str, Any]:
        """Load previous progress if available"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading progress: {e}")
        return {}
    
    def load_existing_pairs(self, session_id: str = None) -> List[Dict[str, Any]]:
        """Load existing DPO pairs from a session file"""
        if session_id is None:
            session_id = self.current_session
        
        session_file = os.path.join(self.output_dir, f"session_{session_id}.jsonl")
        pairs = []
        
        try:
            if os.path.exists(session_file):
                with open(session_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            pair = json.loads(line.strip())
                            pairs.append(pair)
                logger.info(f"Loaded {len(pairs)} existing DPO pairs from session {session_id}")
        except Exception as e:
            logger.error(f"Error loading existing pairs: {e}")
        
        return pairs
    
    def process_recipe_batch_incremental(self, recipes: List[Dict[str, Any]], use_all_questions: bool = True, resume: bool = True) -> str:
        """Process recipes with incremental saving and progress tracking using pre-defined questions"""
        start_time = time.time()
        total_recipes = len(recipes)
        successful_pairs = 0
        failed_pairs = 0
        
        # Check for existing progress
        start_recipe_idx = 0
        if resume:
            progress = self.load_progress()
            if progress and input(f"Resume from {progress.get('completion_percentage', 0):.1f}%? (y/n): ").lower() == 'y':
                start_recipe_idx = progress.get('current_recipe_idx', 0)
                logger.info(f"🔄 Resuming from recipe {start_recipe_idx + 1}/{total_recipes}")
        
        logger.info(f"🚀 Starting DPO generation for {total_recipes} recipes using pre-defined questions")
        logger.info(f"📁 Output file: {self.session_file}")
        
        for recipe_idx, recipe in enumerate(recipes[start_recipe_idx:], start_recipe_idx):
            recipe_start_time = time.time()
            logger.info(f"\n📖 Processing recipe {recipe_idx + 1}/{total_recipes}: {recipe['nombre']}")
            
            # Get all pre-defined questions for this recipe
            # Get questions for this recipe
            questions = self.get_recipe_questions(recipe)
            
            for question_idx, (question, category, context) in enumerate(questions):
                try:
                    # Save progress
                    self.save_progress(recipe_idx, total_recipes, question_idx, len(questions), recipe['nombre'])
                    
                    logger.info(f"  ⚡ Processing Q{question_idx + 1}/{len(questions)} - {category}")
                    
                    # Generate DPO pair
                    dpo_pair = self.generate_dpo_pair(recipe, question, category, context)
                    
                    # Save immediately
                    if self.save_dpo_pair_incremental(dpo_pair):
                        successful_pairs += 1
                    else:
                        failed_pairs += 1
                        logger.warning(f"  ❌ Failed to save Q{question_idx + 1}")
                    
                    # Brief pause to avoid rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    failed_pairs += 1
                    logger.error(f"  ❌ Error processing Q{question_idx + 1}: {e}")
                    continue
            
            recipe_time = time.time() - recipe_start_time
            logger.info(f"  ✅ Completed recipe in {recipe_time:.1f}s - Success: {successful_pairs}, Failed: {failed_pairs}")
        
        # Final summary
        total_time = time.time() - start_time
        logger.info(f"\n🎉 Batch processing complete!")
        logger.info(f"⏱️  Total time: {total_time:.1f}s")
        logger.info(f"✅ Successful pairs: {successful_pairs}")
        logger.info(f"❌ Failed pairs: {failed_pairs}")
        logger.info(f"📁 Output saved to: {self.session_file}")
        
        return self.session_file
    
    def convert_jsonl_to_json(self, session_id: str = None, output_filename: str = None) -> str:
        """Convert JSONL session file to final JSON dataset format"""
        if session_id is None:
            session_id = self.current_session
        
        if output_filename is None:
            output_filename = f"recipes_dpo_{session_id}.json"
        
        session_file = os.path.join(self.output_dir, f"session_{session_id}.jsonl")
        output_path = os.path.join(self.output_dir, output_filename)
        
        pairs = []
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pair = json.loads(line.strip())
                        # Remove session metadata for final dataset
                        clean_pair = {
                            "messages": pair["messages"],
                            "chosen": pair["chosen"],
                            "rejected": pair["rejected"],
                            "metadata": pair["metadata"]
                        }
                        pairs.append(clean_pair)
            
            # Save final JSON format
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(pairs, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📄 Converted {len(pairs)} pairs to final dataset: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting JSONL to JSON: {e}")
            return ""

    def get_session_stats(self, session_id: str = None) -> Dict[str, Any]:
        """Get statistics for a session"""
        if session_id is None:
            session_id = self.current_session

        pairs = self.load_existing_pairs(session_id)

        if not pairs:
            return {
                "total_pairs": 0,
                "categories": {},
                "recipes": {},
                "session_id": session_id,
                "unique_recipes": 0,
                "unique_categories": 0
            }

        # Calculate statistics
        categories = defaultdict(int)
        recipes = defaultdict(int)

        for pair in pairs:
            metadata = pair.get("metadata", {})
            categories[metadata.get("category", "unknown")] += 1
            recipes[metadata.get("recipe_name", "unknown")] += 1

        return {
            "total_pairs": len(pairs),
            "categories": dict(categories),
            "recipes": dict(recipes),
            "session_id": session_id,
            "unique_recipes": len(recipes),
            "unique_categories": len(categories)
        }

    def list_sessions(self) -> List[str]:
        """List all available session IDs"""
        sessions = []
        try:
            for filename in os.listdir(self.output_dir):
                if filename.startswith("session_") and filename.endswith(".jsonl"):
                    session_id = filename.replace("session_", "").replace(".jsonl", "")
                    sessions.append(session_id)
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")

        return sorted(sessions, reverse=True)  # Most recent first

    def validate_dpo_pair(self, pair: Dict[str, Any]) -> bool:
        """Validate a DPO pair meets quality standards"""
        try:
            # Check required fields
            required_fields = ["messages", "chosen", "rejected", "metadata"]
            if not all(field in pair for field in required_fields):
                return False

            # Check message structure
            messages = pair["messages"]
            if len(messages) != 2 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
                return False

            # Check response quality (basic checks)
            chosen = pair["chosen"]
            rejected = pair["rejected"]

            if len(chosen) < 50 or len(rejected) < 20:  # Minimum length check
                return False

            if chosen == rejected:  # Responses shouldn't be identical
                return False

            # Check metadata completeness
            metadata = pair["metadata"]
            required_metadata = ["recipe_id", "recipe_name", "category", "context"]
            if not all(field in metadata for field in required_metadata):
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating DPO pair: {e}")
            return False

    def save_dataset(self, dpo_pairs: List[Dict[str, Any]], filename: str = "ecuadorian_recipes_dpo.json"):
        """Save DPO dataset to JSON file with validation (legacy method)"""

        # Validate all pairs
        valid_pairs = []
        for pair in dpo_pairs:
            if self.validate_dpo_pair(pair):
                valid_pairs.append(pair)
            else:
                logger.warning("Skipping invalid DPO pair")

        # Save to file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(valid_pairs, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(valid_pairs)} valid DPO pairs to {output_path}")
        return len(valid_pairs)

    def generate_complete_dataset(self, recipes: List[Dict[str, Any]], use_all_questions: bool = True, resume: bool = True) -> str:
        """Complete workflow: generate DPO pairs using pre-defined questions with incremental saving"""
        
        logger.info(f"🎯 Starting complete DPO dataset generation using pre-defined questions")

        # Calculate estimated pairs based on questions bank
        total_estimated_pairs = 0
        for recipe in recipes:
            recipe_questions = self.get_recipe_questions(recipe)
            total_estimated_pairs += len(recipe_questions)

        logger.info(f"📊 {len(recipes)} recipes with ~{total_estimated_pairs} pre-defined questions")

        # Process with incremental saving
        session_file = self.process_recipe_batch_incremental(recipes, use_all_questions, resume)

        if not session_file:
            logger.error("❌ Failed to generate DPO pairs")
            return ""

        # Show session statistics
        stats = self.get_session_stats()
        logger.info(f"\n📈 Session Statistics:")
        logger.info(f"   Total pairs: {stats['total_pairs']}")
        logger.info(f"   Unique recipes: {stats['unique_recipes']}")
        logger.info(f"   Categories: {list(stats['categories'].keys())}")

        # Convert to final JSON format
        final_dataset = self.convert_jsonl_to_json()

        logger.info(f"🎉 Complete dataset ready: {final_dataset}")
        return final_dataset

# Example usage and testing
def main():
    """Example usage of the DPO generator with pre-defined questions"""

    print("DPO Generator with Pre-defined Questions")
    print("=" * 60)

    # Initialize the generator with questions file
    generator = GastronomiaDPOGenerator(
        cohere_api_key="COHERE_API_KEY",  # Replace with your actual API key
        questions_file="recipe_questions.json",
        output_dir="dpo_output"
    )

    # Load actual recipe data
    try:
        recipes_json_path = "somosnpl-recetas-zero.json"
        print(f"\n📂 Loading recipes from: {recipes_json_path}")

        with open(recipes_json_path, 'r', encoding='utf-8') as f:
            recipes = json.load(f)

        print(f"✅ Loaded {len(recipes)} recipes")

        # Show first few recipe IDs for debugging
        recipe_ids = [recipe.get('id') for recipe in recipes[:5]]
        print(f"🔍 First 5 recipe IDs: {recipe_ids}")

    except FileNotFoundError:
        print(f"❌ Recipe file not found: {recipes_json_path}")

    # Test with first recipe to debug
    if recipes:
        print(f"\n🧪 Testing with first recipe: {recipes[0].get('nombre', 'Unknown')} (ID: {recipes[0].get('id')})")
        test_questions = generator.get_recipe_questions(recipes[0])
        print(f"📋 Found {len(test_questions)} questions for this recipe")

        if test_questions:
            print("✅ Questions found! Proceeding with full dataset generation...")
            # Generate dataset using all pre-defined questions
            final_dataset = generator.generate_complete_dataset(recipes, use_all_questions=True)
        else:
            print("❌ No questions found for test recipe!")
            print("💡 Check that:")
            print("   1. recipe_questions.json exists and contains questions")
            print("   2. Recipe IDs in questions file match recipe IDs in recipe data")
            print("   3. Questions have the correct format with recipe_id field")
    else:
        print("❌ No recipes loaded!")

if __name__ == "__main__":
    main()
