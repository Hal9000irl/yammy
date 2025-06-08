import unittest
import os
import sys
from unittest.mock import patch

# Add project root to sys.path to allow importing main
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from main import load_config # Assuming main.py is in PROJECT_ROOT

class TestConfigLoading(unittest.TestCase):

    def setUp(self):
        self.tests_dir = SCRIPT_DIR
        self.valid_config_path = os.path.join(self.tests_dir, "mock_config_valid.yml")
        self.invalid_yaml_path = os.path.join(self.tests_dir, "mock_config_invalid_yaml.yml")
        self.non_existent_path = os.path.join(self.tests_dir, "non_existent_config.yml")

        # Ensure mock files are as expected for tests
        # Note: The create_file_with_block tool already created these.
        # Re-writing them here ensures they have the exact content expected by the test,
        # overriding any potential cached or previous versions if the tests were run multiple times.
        with open(self.valid_config_path, 'w') as f:
            f.write("application:\n  log_level: \"DEBUG\"\nrasa_service:\n  server_url: \"http://localhost:5005\"\n")
        with open(self.invalid_yaml_path, 'w') as f:
            f.write("application:\n  log_level: \"DEBUG\"\nrasa_service:\n  server_url: \"http://localhost:5005\n# Missing quote")


    def tearDown(self):
        # Clean up mock files
        if os.path.exists(self.valid_config_path):
            os.remove(self.valid_config_path)
        if os.path.exists(self.invalid_yaml_path):
            os.remove(self.invalid_yaml_path)

    def test_load_valid_config(self):
        config = load_config(config_path=self.valid_config_path)
        self.assertIsNotNone(config)
        self.assertEqual(config['application']['log_level'], 'DEBUG')
        self.assertEqual(config['rasa_service']['server_url'], 'http://localhost:5005')

    @patch('builtins.print') # Mock print to check error messages
    def test_load_non_existent_config(self, mock_print):
        config = load_config(config_path=self.non_existent_path)
        self.assertIsNone(config)
        mock_print.assert_any_call(f"ERROR: Configuration file not found at {self.non_existent_path}. Please create it.")

    @patch('builtins.print') # Mock print
    def test_load_invalid_yaml_config(self, mock_print):
        config = load_config(config_path=self.invalid_yaml_path)
        self.assertIsNone(config)
        # The exact error message for invalid YAML might vary slightly based on the PyYAML version and specific error.
        # We will check if *any* part of the expected error message is present.
        found_error_message = False
        for call_args in mock_print.call_args_list:
            if f"ERROR: Could not load or parse configuration file {self.invalid_yaml_path}" in call_args[0][0] and "Unexpected end of stream" in call_args[0][0]:
                 found_error_message = True
                 break
        self.assertTrue(found_error_message, msg=f"Expected error message for invalid YAML not found in print calls: {mock_print.call_args_list}")


if __name__ == '__main__':
    unittest.main()
