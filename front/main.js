let app = new Vue({
  el: '#app',
  data: {
    image: '',
    results_classification: [],
    results_rekognition: [],
  },
  methods: {
    onFileChange: function(e) {
      let files = e.target.files || e.dataTransfer.files;
      if (!files.length)
        return;

      this.showImage(files[0]);
      this.results_classification = [];
      this.results_rekognition = [];
    },
    showImage: function(file) {
      let freader = new FileReader();

      freader.onload = (e) => {
        this.image = e.target.result;
      };
      freader.readAsDataURL(file);
    },
    uploadImage: function(e) {
      let config = {
        headers: {
          'content-type': 'application/octet-stream',
        }
      };

      axios
        .post(
          "https://<<YOUR ENDPOINT URL>>/api/classification",
          this.image,
          config
        )
        .then(response => {
            this.results_classification = response.data.split(",");
        })
        .catch(error => {
            console.log(error);
        });

      axios
        .post(
          "https://<<YOUR ENDPOINT URL>>/api/rekognition",
          this.image,
          config
        )
        .then(response => {
            this.results_rekognition = response.data.split(",");
        })
        .catch(error => {
            console.log(error);
        });
    }
  }
})